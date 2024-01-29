import numpy as np
from math import log, pi
import torch
from torch.distributions import Normal


LOG_PI = log(pi)
EPS_FLOAT = 1e-6
EPS_DOUBLE = 1e-10


def nump(Tensor):
    return Tensor.detach().cpu().numpy()


def get_tensor_type(device, double):
    if device == 'cpu':
        if double:
            return torch.DoubleTensor
        else:
            return torch.FloatTensor
    elif device == 'cuda':
        if double:
            return torch.cuda.DoubleTensor
        else:
            return torch.cuda.FloatTensor


def log_likelihood(Y, C, U, B, noise_prec, c=1.0):
    if noise_prec.ndim == 0:
        noise_prec = noise_prec.repeat(*Y.shape)
    elif noise_prec.ndim == 1:
        assert noise_prec.shape[0] == Y.shape[0]
        # repeat along #samples
        noise_prec = torch.repeat_interleave(noise_prec.unsqueeze(-1), Y.shape[1], dim=1)
    else:
        assert noise_prec.shape == Y.shape
    likelihood = Normal(loc=c * C @ U @ B, scale=noise_prec.rsqrt())
    return likelihood.log_prob(Y).sum()


def log_prior(M, delta_M, mean_M=None, positive=False):
    if delta_M.ndim == 0:
        delta_M = delta_M.repeat(*M.shape)
    elif delta_M.ndim == 1:
        assert M.shape[0] != M.shape[1], 'cannot differentiate between axes, pass reshaped delta'
        if len(delta_M) == M.shape[0]:
            delta_M = torch.repeat_interleave(delta_M.unsqueeze(-1), M.shape[1], dim=1)
        elif len(delta_M) == M.shape[1]:
            delta_M = torch.repeat_interleave(delta_M.unsqueeze(0), M.shape[0], dim=0)
        else:
            raise ValueError('Invalid shape for delta_M')
    else:
        assert delta_M.shape == M.shape

    if mean_M is None:
        mean_M = torch.zeros_like(M)
    else:
        assert mean_M.shape == M.shape
    prior = Normal(loc=mean_M, scale=delta_M.rsqrt())
    if positive:
        # Assume all values in M are positive and fold at 0 --> 2 * pdf --> log(2) + log pdf
        offset = np.log(2) * np.prod(M.shape)
    else:
        offset = 0
    return prior.log_prob(M).sum() + offset


def log_joint(Y, C, U, B, noise_prec, delta_U, delta_B, positive_U=False):
    log_lik = log_likelihood(Y, C, U, B, noise_prec)
    log_prior_U = log_prior(U, delta_U, positive=positive_U)
    log_prior_B = log_prior(B, delta_B)
    return log_lik + log_prior_U + log_prior_B


def hessian_diag(M_left, M_right, noise_prec, delta):
    """Hessian of noise_prec |Y-M_left X M_right|_F^2 + delta |X|_F^2
    delta and noise_prec can also be vector or matrix variate,
    in this case the factors are pulled into the norms.
    The Hessian is given by
        noise_prec * (M_left.T M_left) \kron (M_right M_right.T) + diag(full_delta)
    """
    if noise_prec.ndim == 0:
        noise_prec = noise_prec.reshape(1, 1)
    elif noise_prec.ndim == 1:
        assert noise_prec.shape[0] == M_left.shape[0]
        # repeat along #samples
        noise_prec = noise_prec.reshape(-1, 1)
    else:
        raise ValueError('Invalid shape of noise_prec', noise_prec.shape)

    left_dim, right_dim = M_left.shape[1], M_right.shape[0]
    if delta.ndim == 0:
        delta = delta.reshape(1, 1)
    elif delta.ndim == 1:
        if len(delta) == left_dim:
            delta = delta.reshape(left_dim, 1)
        elif len(delta) == right_dim:
            delta = delta.reshape(1, right_dim)
    else:
        assert delta.shape == torch.Size([left_dim, right_dim])

    # For reference what the einsum op does:
    # diag_mleft2 = ((M_left * noise_prec).T @ M_left).diagonal()
    # diag_mright2 = (M_right @ M_right.T).diagonal()
    diag_mleft2 = torch.einsum('mn,mn->n', M_left * noise_prec, M_left)
    diag_mright2 = torch.einsum('nm,nm->n', M_right, M_right)
    log_lik_hess_diag = torch.outer(diag_mleft2, diag_mright2)
    assert log_lik_hess_diag.shape == torch.Size([left_dim, right_dim])
    log_prior_hess_diag = torch.ones_like(log_lik_hess_diag) * delta
    log_loss_hess_diag = log_lik_hess_diag + log_prior_hess_diag
    return log_loss_hess_diag.flatten()


def hessian_full(M_left, M_right, noise_prec, delta):
    """Hessian of noise_prec|Y-M_left X M_right|_F^2 + delta |X|_F^2
    noise_prec can also be vector or matrix variate, but delta has to be scalar.
    in this case the factors are pulled into the norms.
    The Hessian is given by
        noise_prec (M_left.T M_left) \kron (M_right M_right.T) + diag(full_delta)
    This is identical to `hessian_kron` in case delta is only scalar (isotropic prior)
    """
    if noise_prec.ndim == 0:
        noise_prec = noise_prec.reshape(1, 1)
    elif noise_prec.ndim == 1:
        assert noise_prec.shape[0] == M_left.shape[0]
        # repeat along #samples
        noise_prec = noise_prec.reshape(-1, 1)
    else:
        raise ValueError('Invalid shape of noise_prec', noise_prec.shape)

    left_dim, right_dim = M_left.shape[1], M_right.shape[0]
    if delta.ndim == 0:
        delta = delta.flatten() * torch.ones(left_dim, right_dim)
    elif delta.ndim == 1:
        if len(delta) == M_left.shape[1]:
            delta = delta.repeat(right_dim, 1).T
            assert delta.shape == torch.Size([left_dim, right_dim])
        elif len(delta) == M_right.shape[0]:
            delta = delta.repeat(left_dim, 1)
            assert delta.shape == torch.Size([left_dim, right_dim])
    else:
        assert delta.shape == torch.Size([left_dim, right_dim])

    mleft2 = ((M_left * noise_prec).T @ M_left)
    mright2 = (M_right @ M_right.T)
    H = torch.kron(mleft2, mright2) + torch.diag(delta.flatten())
    return H

    
def log_marginal_likelihood(Y, C, U, B, noise_prec, delta_U, delta_B, diag_hess_U, diag_hess_B, positive_U=False, ignore_U=False):
    """LML = log joint + P/2 log (2pi) - log det (H + Delta)"""
    log_loss = log_joint(Y, C, U, B, noise_prec, delta_U, delta_B, positive_U=positive_U)
    params_U = np.prod(U.shape) if not ignore_U else 0
    params_B = np.prod(B.shape)
    params = (params_U + params_B) / 2 * LOG_PI

    if ignore_U:
        log_det_hess_U = 0
    else:
        if diag_hess_U:
            hess_U = hessian_diag(C, B, noise_prec, delta_U)
            log_det_hess_U = hess_U.log().sum()
        else:
            hess_U = hessian_full(C, B, noise_prec, delta_U)
            log_det_hess_U = hess_U.logdet()

    M_left = C @ U
    n_samples = B.shape[1]
    if diag_hess_B:
        diag_mleft2 = torch.einsum('mn,mn->n', M_left * noise_prec.reshape(-1, 1), M_left) + delta_B
        log_det_hess_B = diag_mleft2.log().sum() * n_samples
    else:
        mleft2 = (M_left * noise_prec.reshape(-1, 1)).T @ M_left + torch.diag(delta_B)
        log_det_hess_B = mleft2.logdet() * n_samples

    # NOTE: if truncation offset shall be added, use the following code and add truncation_offsets to the return
    # if positive_U:
    #     dist = Normal(loc=U.flatten(), scale=torch.rsqrt(log_loss_hess_diag_U))
    #     truncation_offset = (1 - dist.cdf(torch.zeros_like(log_loss_hess_diag_U))).log().sum()
    log_dets = 0.5 * (log_det_hess_U + log_det_hess_B)
    return log_loss + params - log_dets


def path_fa(
    Y, C, n_latents, n_epochs=100, lr=1e-1, 
    positive_U=True, diag_hess_U=False, diag_hess_B=False,
    device='cpu', double=True, empirical_bayes=True, svd_init=True
):
    EPS = EPS_DOUBLE if double else EPS_FLOAT
    def proj(x):
        return torch.clamp(torch.nan_to_num(x), min=EPS, max=1/EPS)

    # extract shapes
    n_samples = Y.shape[1]
    n_pathways = C.shape[1]

    # move to right device and datatype
    tp = get_tensor_type(device, double)
    if not torch.is_tensor(Y):
        Y = torch.from_numpy(Y)
        C = torch.from_numpy(C)
    Y, C = Y.type(tp), C.type(tp)

    # initialise parameters 
    if svd_init and (n_latents < Y.shape[1]):
        _, s, Bt = torch.svd(Y)
        B = (torch.diag(s[:n_latents]) @ Bt[:, :n_latents].T).type(tp)
        weight = s[n_latents]
    else:
        B = torch.randn(n_latents, n_samples).type(tp)
        s = weight = 0.1
    U = torch.zeros(n_pathways, n_latents).type(tp)
    U.requires_grad = True
    B.requires_grad = True

    # initialise hyperparameters
    noise_prec = torch.ones(Y.shape[0]).type(tp)  # per marker
    delta_U = torch.ones_like(U).type(tp)  # for each pathway-latent combination
    delta_B = torch.ones(B.shape[0]).type(tp) * weight  # per latent

    def zero_grad():
        if U.grad is not None:
            U.grad.zero_()
        if B.grad is not None:
            B.grad.zero_()
        if noise_prec.grad is not None:
            noise_prec.grad.zero_()

    losses = list()
    neg_margliks = list()
    for i in range(n_epochs):
        ### Alternatingly update U and B (ALS step).
        # 1. compute loss (- log_joint) and gradient w.r.t. U/B using autodiff
        # 2. compute hessian (diagonal or full) manually for max efficiency
        # 3. update respecting the Hessian structure
        zero_grad()

        if i % 2 == 0:  # update U
            loss = - log_joint(Y, C, U, B.detach(), noise_prec, delta_U, delta_B, positive_U=positive_U)
            loss.backward()
            grad_U = U.grad.data

            if diag_hess_U:
                hess_U = hessian_diag(C, B.detach(), noise_prec, delta_U)
                U.data -= lr * grad_U / hess_U.reshape(U.shape)
            else:
                # Least-squares == Hessian\inv grad
                hess_U = hessian_full(C, B.detach(), noise_prec, delta_U)
                hess_U_chol = torch.linalg.cholesky(hess_U)
                dirc_U = torch.cholesky_solve(grad_U.flatten().unsqueeze(1), hess_U_chol).squeeze()
                U.data -= lr * dirc_U.reshape(U.shape)
            
            if positive_U:
                U.data.clamp_(min=0)

        else:  # update B
            loss = - log_joint(Y, C, U.detach(), B, noise_prec, delta_U, delta_B, positive_U=positive_U)
            loss.backward()
            grad_B = B.grad.data 
            M_left = C @ U.detach()

            if diag_hess_B:
                # the following update is equivalent to computing the Hessian like above but with an unnecessary identity:
                # hess_B = hessian_diag(M_left, torch.eye(n_latents), noise_prec, delta_B)
                diag_mleft2 = torch.einsum('mn,mn->n', M_left * noise_prec.reshape(-1, 1), M_left).unsqueeze(1) + delta_B.reshape(-1, 1)
                B.data -= lr * (grad_B / diag_mleft2)
            else:
                # the following update is equivalent to computing the Hessian like above, again the identity is not necessary
                # hess_B = hessian_full(M_left, torch.eye(n_samples).type(tp), noise_prec, delta_B)
                # B.data -= lr * (hess_B.inverse() @ grad_B.flatten()).reshape(B.shape)
                # efficiently, only need to left-invert since right factor is simply identity
                hess_Bl = ((M_left * noise_prec.reshape(-1, 1)).T @ M_left) + torch.diag(delta_B)
                hess_Bl_chol = torch.linalg.cholesky(hess_Bl)
                dirc_B = torch.cholesky_solve(grad_B, hess_Bl_chol)
                B.data -= lr * dirc_B 

        losses.append(loss.item())
        if not empirical_bayes:
            Y_pred = C @ U.detach() @ B.detach()
            noise_prec = 1 / (Y_pred - Y).square().mean(1)
            continue

        ### Compute Marginal-Likelihood updates for hyperparameters
        neg_marglik = - log_marginal_likelihood(
            Y, C, U.detach(), B.detach(), noise_prec, delta_U, delta_B, 
            diag_hess_U=diag_hess_U, diag_hess_B=diag_hess_B, positive_U=positive_U
        )

        # Update U's prior precision (delta_U)
        if diag_hess_U:
            hess_U = hessian_diag(C, B.detach(), noise_prec, delta_U)
            gamma_U = 1 - delta_U / hess_U.reshape(delta_U.shape)
        else:
            hess_U = hessian_full(C, B.detach(), noise_prec, delta_U)
            gamma_U = 1 - delta_U * hess_U.inverse().diagonal().reshape(delta_U.shape)
        delta_U = proj(gamma_U / U.detach().square())

        # Update B's prior precision (delta_B)
        M_left = C @ U.detach()
        if diag_hess_B:
            diag_mleft2 = torch.einsum('mn,mn->n', M_left * noise_prec.reshape(-1, 1), M_left) + delta_B
            gamma_B = (1 - delta_B / diag_mleft2) * B.shape[1]
        else:
            hess_Bl = ((M_left * noise_prec.reshape(-1, 1)).T @ M_left) + torch.diag(delta_B)
            gamma_B = (1 - delta_B * hess_Bl.inverse().diagonal()) * B.shape[1]
        delta_B = proj(gamma_B / B.detach().square().sum(1))

        # Update noise precision (inverse of observation variance)
        # 1. compute gradient w.r.t. logdet using autograd
        noise_prec.requires_grad = True
        if diag_hess_U:
            log_det_hess_U = hessian_diag(C, B.detach(), noise_prec, delta_U).log().sum()
        else:
            log_det_hess_U = hessian_full(C, B.detach(), noise_prec, delta_U).logdet()
        
        if diag_hess_B:
            diag_mleft2 = torch.einsum('mn,mn->n', M_left * noise_prec.reshape(-1, 1), M_left) + delta_B
            log_det_hess_B = n_samples * diag_mleft2.log().sum()
        else:
            hess_Bl = ((M_left * noise_prec.reshape(-1, 1)).T @ M_left) + torch.diag(delta_B)
            log_det_hess_B = n_samples * hess_Bl.logdet()
        (log_det_hess_U + log_det_hess_B).backward()
        errs = (Y - C @ U.detach() @ B.detach()).square().sum(1)
        beta_grad = noise_prec.grad
        traces = beta_grad * noise_prec.data
        noise_prec = proj((n_samples - traces) / errs)
        noise_prec.requires_grad = False

        neg_margliks.append(neg_marglik.item())

    params = {'U': nump(U), 'B': nump(B), 'sigma_noise': nump(noise_prec.rsqrt()),
              'delta_U': nump(delta_U), 'delta_B': nump(delta_B)}
    return params, losses, neg_margliks


def log_multi_joint(Yr, Cr, Yp, Cp, U, B, noise_prec_r, noise_prec_p,
                    delta_U, delta_B, cr=1.0, cp=1.0, positive_U=False):
    log_lik_r = log_likelihood(Yr, Cr, U, B, noise_prec_r, c=cr)
    log_lik_p = log_likelihood(Yp, Cp, U, B, noise_prec_p, c=cp)
    log_prior_U = log_prior(U, delta_U, positive=positive_U)
    log_prior_B = log_prior(B, delta_B)
    return log_lik_r + log_lik_p + log_prior_U + log_prior_B

    
def log_multi_marginal_likelihood(Yr, Cr, Yp, Cp, U, B, noise_prec_r, noise_prec_p, delta_U, delta_B,
                                  cr=1.0, cp=1.0, diag_hess_U=False, diag_hess_B=False, positive_U=False, ignore_U=False):
    log_loss = log_multi_joint(Yr, Cr, Yp, Cp, U, B, noise_prec_r, noise_prec_p, delta_U, delta_B, 
                               cr, cp, positive_U)
    params_U = np.prod(U.shape) if not ignore_U else 0
    params_B = np.prod(B.shape)
    params = (params_U + params_B) / 2 * LOG_PI

    if ignore_U:
        log_det_hess_U = 0
    else:
        if diag_hess_U:
            hess_U = (hessian_diag(cr * Cr, B, noise_prec_r, delta_U)
                    + hessian_diag(cp * Cp, B, noise_prec_p, torch.zeros_like(delta_U)))
            log_det_hess_U = hess_U.log().sum()
        else:
            hess_U = (hessian_full(cr * Cr, B, noise_prec_r, delta_U)
                    + hessian_full(cp * Cp, B, noise_prec_p, torch.zeros_like(delta_U)))
            log_det_hess_U = hess_U.logdet()
    
    M_left_r = cr * Cr @ U
    M_left_p = cp * Cp @ U
    n_samples = B.shape[1]
    if diag_hess_B:
        diag_mleft2 = (torch.einsum('mn,mn->n', M_left_r * noise_prec_r.reshape(-1, 1), M_left_r) 
                       + torch.einsum('mn,mn->n', M_left_p * noise_prec_p.reshape(-1, 1), M_left_p) 
                       + delta_B)
        log_det_hess_B = diag_mleft2.log().sum() * n_samples
    else:
        mleft2 = (((M_left_r * noise_prec_r.reshape(-1, 1)).T @ M_left_r) 
                  + ((M_left_p * noise_prec_p.reshape(-1, 1)).T @ M_left_p)
                  + torch.diag(delta_B))
        log_det_hess_B = mleft2.logdet() * n_samples

    log_dets = 0.5 * (log_det_hess_U + log_det_hess_B)
    return log_loss + params - log_dets


def multi_path_fa(
    Yr, Cr, Yp, Cp, n_latents, n_epochs=100, lr=1e-1,
    positive_U=True, diag_hess_U=False, diag_hess_B=False,
    device='cpu', double=True, factor='scalar',
    empirical_bayes=True, svd_init=True
):
    EPS = EPS_DOUBLE if double else EPS_FLOAT
    def proj(x):
        return torch.clamp(torch.nan_to_num(x), min=EPS, max=1/EPS)

    # extract shapes
    n_samples = Yr.shape[1]
    n_pathways = Cr.shape[1]
    n_markers_r = Yr.shape[0]
    n_markers_p = Yp.shape[0]
    assert Cp.shape[1] == n_pathways
    assert Yp.shape[1] == n_samples

    # move to right device and datatype
    tp = get_tensor_type(device, double)
    if not torch.is_tensor(Yr):
        Yr = torch.from_numpy(Yr)
        Cr = torch.from_numpy(Cr)
        Yp = torch.from_numpy(Yp)
        Cp = torch.from_numpy(Cp)
    Yr, Cr = Yr.type(tp), Cr.type(tp)
    Yp, Cp = Yp.type(tp), Cp.type(tp)

    # initialise parameters
    if svd_init and (n_latents < Yr.shape[1]):
        _, s, Bt = torch.svd(Yr)
        B = (torch.diag(s[:n_latents]) @ Bt[:, :n_latents].T).type(tp)
    else:
        B = torch.randn(n_latents, n_samples).type(tp)
    U = torch.zeros(n_pathways, n_latents).type(tp)
    U.requires_grad = True
    B.requires_grad = True
    if factor == 'scalar':
        cr = torch.ones((1,)).type(tp)
        cp = torch.ones((1,)).type(tp)
        cr.requires_grad = True
        cp.requires_grad = True
    elif factor == 'vector':
        cr = torch.ones((n_markers_r, 1)).type(tp)
        cp = torch.ones((n_markers_p, 1)).type(tp)
        cr.requires_grad = True
        cp.requires_grad = True
    elif factor == 'none':
        cr = cp = 1.0
    else:
        raise ValueError(f'Invalid factor: {factor}')

    # initialise hyperparameters
    noise_prec_r = torch.ones(n_markers_r).type(tp)  # per marker
    noise_prec_p = torch.ones(n_markers_p).type(tp) # per marker
    latent_order = torch.exp(torch.linspace(-4, 4, B.shape[0]))  # prefer ordering
    delta_U = torch.ones_like(U).type(tp) * latent_order.unsqueeze(0) # for each pathway-latent
    delta_B = torch.ones(B.shape[0]).type(tp) * latent_order # per latent

    def zero_grad():
        if U.grad is not None:
            U.grad.zero_()
        if B.grad is not None:
            B.grad.zero_()
        if noise_prec_r.grad is not None:
            noise_prec_r.grad.zero_()
        if noise_prec_p.grad is not None:
            noise_prec_p.grad.zero_()

    losses = list()
    neg_margliks = list()
    for i in range(n_epochs):
        # param step
        zero_grad()
        if factor != 'none':
            if cr.grad is not None:
                cr.grad.zero_()
                cp.grad.zero_()
            cr.requires_grad = True
            cp.requires_grad = True

        if i % 2 == 0:  # update U
            loss = - log_multi_joint(Yr, Cr, Yp, Cp, U, B.detach(), noise_prec_r, noise_prec_p, delta_U, delta_B,
                                     cr, cp, positive_U=positive_U)
            loss.backward()
            grad_U = U.grad.data

            if diag_hess_U:
                hess_U = (hessian_diag(cr * Cr, B.detach(), noise_prec_r, delta_U)
                          + hessian_diag(cp * Cp, B.detach(), noise_prec_p, torch.zeros_like(delta_U)))
                U.data -= lr * grad_U / hess_U.reshape(U.shape)
            else:
                # Least-squares == Hessian\inv grad
                hess_U = (hessian_full(cr * Cr, B.detach(), noise_prec_r, delta_U)
                          + hessian_full(cp * Cp, B.detach(), noise_prec_p, torch.zeros_like(delta_U)))
                hess_U_chol = torch.linalg.cholesky(hess_U)
                dirc_U = torch.cholesky_solve(grad_U.flatten().unsqueeze(1), hess_U_chol).squeeze()
                U.data -= lr * dirc_U.reshape(U.shape)
            
            if positive_U:
                U.data.clamp_(min=0)

        else:  # update B
            loss = - log_multi_joint(Yr, Cr, Yp, Cp, U.detach(), B, noise_prec_r, noise_prec_p, delta_U, delta_B,
                                     cr, cp, positive_U=positive_U)
            loss.backward()
            grad_B = B.grad.data 

            M_left_r = cr * Cr @ U.detach()
            M_left_p = cp * Cp @ U.detach()
            if diag_hess_B:
                diag_mleft2 = (torch.einsum('mn,mn->n', M_left_r * noise_prec_r.reshape(-1, 1), M_left_r) 
                               + torch.einsum('mn,mn->n', M_left_p * noise_prec_p.reshape(-1, 1), M_left_p)
                               + delta_B).reshape(-1, 1)
                B.data -= lr * (grad_B / diag_mleft2)
            else:
                hess_Bl = ((M_left_r * noise_prec_r.reshape(-1, 1)).T @ M_left_r
                           + (M_left_p * noise_prec_p.reshape(-1, 1)).T @ M_left_p
                           + torch.diag(delta_B))
                hess_Bl_chol = torch.linalg.cholesky(hess_Bl)
                dirc_B = torch.cholesky_solve(grad_B, hess_Bl_chol)
                B.data -= lr * dirc_B 

        if factor != 'none':
            Z = U.detach() @ B.detach()
            m_right_r = Cr @ Z
            m_right_p = Cp @ Z
            if factor == 'scalar':
                hess_cr = (m_right_r.square() * noise_prec_r.reshape(-1, 1)).sum() + EPS
                hess_cp = (m_right_p.square() * noise_prec_p.reshape(-1, 1)).sum() + EPS
            elif factor == 'vector':
                hess_cr = (m_right_r.square() * noise_prec_r.reshape(-1, 1)).sum(dim=1).unsqueeze(1) + EPS
                hess_cp = (m_right_p.square() * noise_prec_p.reshape(-1, 1)).sum(dim=1).unsqueeze(1) + EPS
            cr.data -= lr * cr.grad.data / hess_cr
            cp.data -= lr * cp.grad.data / hess_cp
            cr.requires_grad = False
            cp.requires_grad = False

        losses.append(loss.item())
        if not empirical_bayes:
            Y_pred_r = cr * Cr @ U.detach() @ B.detach()
            Y_pred_p = cp * Cp @ U.detach() @ B.detach()
            s2_r = (Y_pred_r - Yr).square().mean(1)
            s2_p = (Y_pred_p - Yp).square().mean(1)
            noise_prec_r = 1 / s2_r
            noise_prec_p = 1 / s2_p
            continue

        ### Compute Marginal-Likelihood updates for hyperparameters
        neg_marglik = - log_multi_marginal_likelihood(
            Yr, Cr, Yp, Cp, U.detach(), B.detach(), noise_prec_r, noise_prec_p, delta_U, delta_B,
            cr, cp, diag_hess_U, diag_hess_B, positive_U
        )

        # Update U's prior precision (delta_U)
        if diag_hess_U:
            hess_U = (hessian_diag(cr * Cr, B.detach(), noise_prec_r, delta_U)
                      + hessian_diag(cp * Cp, B.detach(), noise_prec_p, torch.zeros_like(delta_U)))
            gamma_U = 1 - delta_U / hess_U.reshape(delta_U.shape)
        else:
            hess_U = (hessian_full(cr * Cr, B.detach(), noise_prec_r, delta_U)
                      + hessian_full(cp * Cp, B.detach(), noise_prec_p, torch.zeros_like(delta_U)))
            gamma_U = 1 - delta_U * hess_U.inverse().diagonal().reshape(delta_U.shape)
        delta_U = proj(gamma_U / U.detach().square())

        # update B's prior precision (delta_B)
        M_left_r = cr * Cr @ U.detach()
        M_left_p = cp * Cp @ U.detach()
        if diag_hess_B:
            diag_mleft2 = (torch.einsum('mn,mn->n', M_left_r * noise_prec_r.reshape(-1, 1), M_left_r) 
                           + torch.einsum('mn,mn->n', M_left_p * noise_prec_p.reshape(-1, 1), M_left_p)
                           + delta_B)
            gamma_B = (1 - delta_B / diag_mleft2) * n_samples
        else:
            hess_Bl = ((M_left_r * noise_prec_r.reshape(-1, 1)).T @ M_left_r
                       + (M_left_p * noise_prec_p.reshape(-1, 1)).T @ M_left_p
                       + torch.diag(delta_B))
            gamma_B = (1 - delta_B * hess_Bl.inverse().diagonal()) * n_samples
        delta_B = proj(gamma_B / B.detach().square().sum(1))

        # Update noise precisions
        noise_prec_r.requires_grad = True
        noise_prec_p.requires_grad = True
        if diag_hess_U:
            hess_U = (hessian_diag(cr * Cr, B, noise_prec_r, delta_U)
                      + hessian_diag(cp * Cp, B, noise_prec_p, torch.zeros_like(delta_U)))
            log_det_hess_U = hess_U.log().sum()
        else:
            hess_U = (hessian_full(cr * Cr, B, noise_prec_r, delta_U)
                      + hessian_full(cp * Cp, B, noise_prec_p, torch.zeros_like(delta_U)))
            log_det_hess_U = hess_U.logdet()

        if diag_hess_B:
            diag_mleft2 = (torch.einsum('mn,mn->n', M_left_r * noise_prec_r.reshape(-1, 1), M_left_r) 
                           + torch.einsum('mn,mn->n', M_left_p * noise_prec_p.reshape(-1, 1), M_left_p) 
                           + delta_B)
            log_det_hess_B = diag_mleft2.log().sum() * n_samples
        else:
            mleft2 = (((M_left_r * noise_prec_r.reshape(-1, 1)).T @ M_left_r) 
                      + ((M_left_p * noise_prec_p.reshape(-1, 1)).T @ M_left_p)
                      + torch.diag(delta_B))
            log_det_hess_B = mleft2.logdet() * n_samples

        (log_det_hess_U + log_det_hess_B).backward()
        Z = U.detach() @ B.detach()
        errs_r = (Yr - cr * Cr @ Z).square().sum(1)
        errs_p = (Yp - cp * Cp @ Z).square().sum(1)
        beta_grad_r = noise_prec_r.grad
        beta_grad_p = noise_prec_p.grad
        traces_r = beta_grad_r * noise_prec_r.data
        traces_p = beta_grad_p * noise_prec_p.data
        noise_prec_r = proj((n_samples - traces_r) / errs_r)
        noise_prec_p = proj((n_samples - traces_p) / errs_p)
        noise_prec_r.requires_grad = False
        noise_prec_p.requires_grad = False

        neg_margliks.append(neg_marglik.item())

    if factor == 'scalar':
        cr, cp = cr.item(), cp.item()
    elif factor == 'vector':
        cr, cp = nump(cr), nump(cp)

    params = {'U': nump(U), 'B': nump(B), 'sigma_noise_r': nump(noise_prec_r.rsqrt()), 'sigma_noise_p': nump(noise_prec_p.rsqrt()),
              'delta_U': nump(delta_U), 'delta_B': nump(delta_B), 'cr': cr, 'cp': cp}
    return params, losses, neg_margliks


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # test standard bayplier methods
    Y = torch.randn(5, 3)
    C = torch.randn(5, 4)
    U = torch.randn(4, 2, requires_grad=True)
    B = torch.randn(2, 3)
    sigma_noise = torch.tensor(0.235)
    delta_C = torch.tensor(1.62)
    delta_U = torch.tensor(0.652)
    delta_U = torch.randn(4, 2) * 5 + 10
    delta_B = torch.tensor(9.23)
    log_loss = - log_joint(Y, C, U, B, sigma_noise, delta_U, delta_B)
    func = lambda t: - log_joint(Y, C, t, B, sigma_noise, delta_U, delta_B)
    true_hess = torch.autograd.functional.hessian(func, U).view(4*2, 4*2).diagonal()
    implemented_hess = hessian_diag(C, B, sigma_noise, delta_U)
    assert torch.allclose(true_hess, implemented_hess)

    # run bayplier for testing
    ps, ls, ng = path_fa(Y.numpy(), C.numpy(), 2, learn_C=False)
    print('results:', ps)
    plt.plot(ls)
    plt.show()
    plt.plot(ng)
    plt.show()

    # test baymultiplier methods
    Yr, Cr = Y, C
    Yp, Cp = torch.randn(7, 3), torch.randn(7, 4)
    sigma_noise_r = sigma_noise
    sigma_noise_p = torch.tensor(0.89)
    U = torch.randn(4, 2, requires_grad=True)
    log_loss = - log_multi_joint(Yr, Cr, Yp, Cp, U, B, sigma_noise_r, sigma_noise_p, delta_U, delta_B)
    func = lambda t: - log_multi_joint(Yr, Cr, Yp, Cp, t, B, sigma_noise_r, sigma_noise_p, delta_U, delta_B)
    true_hess = torch.autograd.functional.hessian(func, U).view(4*2, 4*2).diagonal()
    implemented_hess = hessian_diag(Cr, B, sigma_noise_r, delta_U)
    implemented_hess += hessian_diag(Cp, B, sigma_noise_p, torch.zeros_like(delta_U))
    assert torch.allclose(true_hess, implemented_hess)

    # run baymultiplier for testing
    ps, ls, ng = multi_path_fa(Yr, Cr, Yp, Cp, 2)
    print('results:', ps)
    plt.plot(ls)
    plt.show()
    plt.plot(ng)
    plt.show()
