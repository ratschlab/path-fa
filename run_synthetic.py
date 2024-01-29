import os, sys
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
from torch.distributions import Normal
from scipy.spatial.distance import cdist
from sklearn.decomposition import FactorAnalysis

from pathfa.path_fa import multi_path_fa, path_fa
from pathfa.plier_R import plier as run_plier_R
from mofapy.core.entry_point import entry_point

EPS = 1e-5  # values below considered zero for sparsity


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


class HiddenPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def read_hallmark_df(datatype='rna'):
    df = pd.read_csv(f'data_synth/msigdb-hallmark_{datatype}.csv', index_col=0)
    def mapper(s):
        mapper_dict = dict(IL2_STAT5_SIGNALING='IL2_JAK_STAT5_SIGNALING',
                           KRAS_SIGNALING_DN='KRAS_SIGNALING_DOWN',
                           PANCREAS_BETA_CELLS='PANCREAS_BETA_CELL',
                           UV_RESPONSE_DN='UV_RESPONSE_DOWN'
        )
        return mapper_dict.get(s, s)
    df.columns = [mapper('_'.join(e.split('_')[1:])) for e in df.columns]
    return df


def generate_synthetic_data(
    n_samples=10000, positive_U=True, seed=711, n_modes=3, B_noise_scale=1.0,
    heteroscedastic_B=True, heteroscedastic_noise=True
):
    np.random.seed(seed)
    df_U = pd.read_csv('data_synth/msigdb-hallmark_U_indices.csv', index_col=0)
    df_C_rna = read_hallmark_df('rna')[df_U.index]
    C_rna = df_C_rna.values.astype(float)
    df_C_prot = read_hallmark_df('prot')[df_U.index]
    C_prot = df_C_prot.values.astype(float)
    U_data = np.random.randn(*df_U.shape)
    if positive_U:
        U_data = np.abs(U_data)
    U = df_U.values.astype(float) * U_data
    # NOTE: noise scales match realistic observations. 
    # RNA: mean = 0.9, std = 1.0
    if heteroscedastic_noise:
        noise_scale_rna = np.exp(np.random.randn(C_rna.shape[0], 1) * 0.11)
        noise_scale_rna = noise_scale_rna / noise_scale_rna.mean() * 0.95
    else:
        noise_scale_rna = np.ones((C_rna.shape[0], 1)) * 0.95
    noise_rna = np.random.randn(C_rna.shape[0], n_samples) * noise_scale_rna
    # Prot: mean = 0.5, std = 0.65
    if heteroscedastic_noise:
        noise_scale_prot = np.exp(np.random.randn(C_prot.shape[0], 1) * 0.06)
        noise_scale_prot = noise_scale_prot / noise_scale_prot.mean() * 0.98
    else:
        noise_scale_prot = np.ones((C_prot.shape[0], 1)) * 0.98
    noise_prot = np.random.randn(C_prot.shape[0], n_samples) * noise_scale_prot
    # just slightly perturb the relevances (can also increase it.. by not dividing by 10)
    if heteroscedastic_B:
        latent_relevances = np.exp(np.random.randn(df_U.shape[1])) * B_noise_scale * 0.5
    else:
        latent_relevances = np.ones(df_U.shape[1]) * B_noise_scale * 0.5
    # sample n_modes (number of expected patient clusters)
    B_modes = np.random.randn(df_U.shape[1], n_modes) * B_noise_scale
    mode = np.random.choice(n_modes, n_samples)
    # B = mode(n_latents x n_samples) + noise(n_latents x n_samples)
    B = B_modes[:, mode] + np.random.randn(df_U.shape[1], n_samples) * latent_relevances[:, np.newaxis]
    Y_rna = C_rna @ U @ B + noise_rna
    Y_prot = C_prot @ U @ B + noise_prot
    observed_rna, observed_prot = (Y_rna.std(1) != 0), (Y_prot.std(1) != 0)
    return (
        Y_rna[observed_rna], C_rna[observed_rna],
        noise_scale_rna[observed_rna],
        Y_prot[observed_prot], C_prot[observed_prot],
        noise_scale_prot[observed_prot],
        U, B
    )


def log_likelihood(Y, C, U, B, noise_prec, c=1.0):
    tt = lambda x: torch.from_numpy(x)
    Y, C, U, B = tt(Y), tt(C), tt(U), tt(B)
    noise_prec = tt(noise_prec).flatten()
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


def compute_multi_recovery(Y_test_rna, Y_test_prot, C_rna, C_prot, U_true, B_true, noise_true_rna, noise_true_prot,  # true
                           U_learned, c_rna, c_prot, sigma_noise_rna, sigma_noise_prot, delta_B):                    # learned
    assert Y_test_rna.shape[1] == Y_test_prot.shape[1]
    noise_prec_r = 1 / np.square(sigma_noise_rna)
    noise_prec_p = 1 / np.square(sigma_noise_prot)
    # compute B_test and compute U_true B_True vs U_learned B_fit
    M_rna = c_rna * C_rna @ U_learned
    M_prot = c_prot * C_prot @ U_learned
    H = (M_rna * noise_prec_r.reshape(-1, 1)).T @ M_rna + (M_prot * noise_prec_p.reshape(-1, 1)).T @ M_prot + np.diag(delta_B)
    grad = (c_rna * (C_rna @ U_learned).T @ (Y_test_rna * noise_prec_r.reshape(-1, 1))
            + c_prot * (C_prot @ U_learned).T @ (Y_test_prot * noise_prec_p.reshape(-1, 1)))
    H_chol = torch.linalg.cholesky(torch.from_numpy(H))
    B_test = torch.cholesky_solve(torch.from_numpy(grad), H_chol).numpy()
    reconst_rna = c_rna * C_rna @ U_learned @ B_test
    reconst_prot = c_prot * C_prot @ U_learned @ B_test
    neg_log_lik_rna = log_likelihood(reconst_rna, C_rna, U_true, B_true, noise_true_rna).item() / np.prod(reconst_rna.shape)
    neg_log_lik_prot = log_likelihood(reconst_prot, C_prot, U_true, B_true, noise_true_prot).item() / np.prod(reconst_prot.shape)
    a, b = (U_true @ B_true), (U_learned @ B_test)
    pathway_residual = np.sqrt(np.sum(np.square(a - b))) ** 2 / a.shape[1]
    return dict(nll_rna=neg_log_lik_rna, nll_prot=neg_log_lik_prot, pathway_l2=pathway_residual)


def compute_single_recovery(Y_test, C, U_true, B_true, noise_true,             # true
                            U_learned, Z_learned, sigma_noise, delta_B,        # learned
                            m=None, s=None):       
    noise_prec = 1 / np.square(sigma_noise)
    if Z_learned is None:
        M = C @ U_learned
    else:
        M = Z_learned
    H = (M * noise_prec.reshape(-1, 1)).T @ M + np.diag(delta_B)
    grad = M.T @ (Y_test * noise_prec.reshape(-1, 1))
    H_chol = torch.linalg.cholesky(torch.from_numpy(H))
    B_test = torch.cholesky_solve(torch.from_numpy(grad), H_chol).numpy()
    reconst = M @ B_test
    if m is not None:
        assert s is not None
        reconst = reconst * s.reshape(-1, 1) + m.reshape(-1, 1) 
    neg_log_lik = log_likelihood(reconst, C, U_true, B_true, noise_true).item() / np.prod(reconst.shape)
    a, b = (U_true @ B_true), (U_learned @ B_test)
    pathway_residual = np.sqrt(np.sum(np.square(a - b))) ** 2 / a.shape[1]
    return dict(nll=neg_log_lik, pathway_l2=pathway_residual)


def compute_multi_performances(Y_test_rna, Y_test_prot, C_rna, C_prot, U_true, B_true, sigma_rna_true, sigma_prot_true, params):
    perf = dict(joint=dict())
    noise_rna = 1 / np.square(sigma_rna_true)
    noise_prot = 1 / np.square(sigma_prot_true)
    sparsity_fraction_l1 = np.abs((np.abs(U_true) > EPS).sum() - (np.abs(params['U']) > EPS).sum()) / np.prod(U_true.shape)
    perf['joint']['sparsity_fraction_l1'] = sparsity_fraction_l1
    if 'sigma_noise_p' in params:
        perf['prot'] = {'sigma_l1': np.mean(np.abs(params['sigma_noise_p'] - sigma_prot_true.ravel()))}
        perf['rna'] = {'sigma_l1': np.mean(np.abs(params['sigma_noise_r'] - sigma_rna_true.ravel()))}
    else:
        params['sigma_noise_p'] = params['sigma_noise_r'] = np.ones(1)
        params['delta_B'] = np.ones(params['U'].shape[1])
        perf['prot'] = dict()
        perf['rna'] = dict()
    # compute performance on joint update
    perf['joint'].update(compute_multi_recovery(
        Y_test_rna, Y_test_prot, C_rna, C_prot, U_true, B_true, noise_rna, noise_prot,
        params['U'], params['cr'], params['cp'], params['sigma_noise_r'], params['sigma_noise_p'], params['delta_B']
    ))
    # compute performance on marginal updates
    perf['rna'].update(compute_single_recovery(
        Y_test_rna, C_rna, U_true, B_true, noise_rna, params['U'], None, params['sigma_noise_r'], params['delta_B']
    ))
    perf['prot'].update(compute_single_recovery(
        Y_test_prot, C_prot, U_true, B_true, noise_prot, params['U'], None, params['sigma_noise_p'], params['delta_B']
    ))
    return perf


def compute_single_performances(Y_test, C, U_true, B_true, sigma_true, params, m=None, s=None):
    perf = dict()
    noise_true = 1 / np.square(sigma_true)
    sparsity_fraction_l1 = np.abs((np.abs(U_true) > EPS).sum() - (np.abs(params['U']) > EPS).sum()) / np.prod(U_true.shape)
    perf['sparsity_fraction_l1'] = sparsity_fraction_l1
    if 'sigma_noise' in params:
        perf['sigma_l1'] = np.mean(np.abs(params['sigma_noise'] - sigma_true.ravel()))
    else:
        params['sigma_noise'] = np.ones(1)
    if 'delta_B' not in params:
        params['delta_B'] = np.ones(params['U'].shape[1])
    if 'Z' not in params:
        params['Z'] = None
    perf.update(compute_single_recovery(
        Y_test, C, U_true, B_true, noise_true, params['U'], params['Z'], params['sigma_noise'], params['delta_B'], m=m, s=s
    ))
    return perf


def main(svd_init=True, seed=711, n_epochs_pathfa=200, n_epochs_mofa=1000, n_epochs_plier=400, 
         n_modes=3, B_noise_scale=1.0, het_B=False, het_noise=True):
    Y_rna, C_rna, sigma_rna, Y_prot, C_prot, sigma_prot, U, B = generate_synthetic_data(
        n_samples=6000, positive_U=True, seed=seed, n_modes=n_modes, B_noise_scale=B_noise_scale,
        heteroscedastic_B=het_B, heteroscedastic_noise=het_noise
    )
    Y_rna_test = Y_rna[:, -5000:]
    Y_prot_test = Y_prot[:, -5000:]
    B_test = B[:, -5000:]

    performance = dict()
    # From 10^1 to 10^3
    for n_subsamples in np.logspace(1, 3, 20).astype(int):
        performance[n_subsamples] = dict()
        print('-------------------', n_subsamples, '------------------')
        Y_rna_sub = Y_rna[:, :n_subsamples]
        Y_prot_sub = Y_prot[:, :n_subsamples]

        # PathFA
        set_seed(seed)
        params, losses, neg_margliks = multi_path_fa(
            Y_rna_sub, C_rna, Y_prot_sub, C_prot, n_latents=U.shape[1], n_epochs=n_epochs_pathfa,
            lr=1e-1, positive_U=True, diag_hess_U=False, diag_hess_B=False, factor='none', 
            empirical_bayes=True, svd_init=svd_init
        )
        perf = compute_multi_performances(Y_rna_test, Y_prot_test, C_rna, C_prot, U, B_test, sigma_rna, sigma_prot, params)
        perf['joint']['nlj'] = losses[-1]
        perf['joint']['nlml'] = neg_margliks[-1]
        performance[n_subsamples]['multipathfa'] = perf

        # MOFA
        with HiddenPrint():
            ep = entry_point()
            data = [Y_rna_sub, Y_prot_sub]
            ep.set_data(data)
            ep.set_model_options(factors=U.shape[1], likelihoods=['gaussian', 'gaussian'], sparsity=True)
            ep.set_data_options(view_names=['rna', 'prot'], RemoveIncompleteSamples=False); ep.parse_data();
            ep.set_train_options(iter=n_epochs_mofa, tolerance=0.01, dropR2=0, elbofreq=1, verbose=False, seed=seed)
            ep.define_priors(); ep.define_init(); ep.parse_intercept(); ep.train_model();
        expectations = ep.model.getExpectations(only_first_moments=True)
        W_rna, W_prot = np.nan_to_num(expectations['SW'][0]), np.nan_to_num(expectations['SW'][1])
        precr = np.nan_to_num(expectations['Tau'][0].T[:, :1], nan=1.0)  # repeated for each sample
        precp = np.nan_to_num(expectations['Tau'][1].T[:, :1], nan=1.0)
        # MOFA uses a Z ~ Normal(0, 1) prior and needs to be updated keeping W's fixed
        H = (W_rna * precr).T @ W_rna + (W_prot * precp).T @ W_prot + np.eye(W_prot.shape[1])
        grad = W_rna.T @ (Y_rna_test * precr) + W_prot.T @ (Y_prot_test * precp)
        H_inv = np.nan_to_num(np.linalg.inv(H))
        Z_test = H_inv @ grad
        # joint reconstruction
        reconst_rna = W_rna @ Z_test
        reconst_prot = W_prot @ Z_test
        neg_log_lik_rna = log_likelihood(reconst_rna, C_rna, U, B_test, 1 / np.square(sigma_rna)).item() / np.prod(reconst_rna.shape)
        neg_log_lik_prot = log_likelihood(reconst_prot, C_prot, U, B_test, 1 / np.square(sigma_prot)).item() / np.prod(reconst_prot.shape)
        # marginal reconstruction
        reconst_rna = W_rna @ H_inv @ W_rna.T @ (Y_rna_test * precr)
        neg_log_lik_rna_marg = log_likelihood(reconst_rna, C_rna, U, B_test, 1 / np.square(sigma_rna)).item() / np.prod(reconst_rna.shape)
        reconst_prot = W_prot @ H_inv @ W_prot.T @ (Y_prot_test * precp)
        neg_log_lik_prot_marg = log_likelihood(reconst_prot, C_prot, U, B_test, 1 / np.square(sigma_prot)).item() / np.prod(reconst_prot.shape)
        perf = dict(joint=dict(nll_rna=neg_log_lik_rna, nll_prot=neg_log_lik_prot), 
                    rna=dict(nll=neg_log_lik_rna_marg),
                    prot=dict(nll=neg_log_lik_prot_marg))
        perf['prot']['sigma_l1'] = np.mean(np.abs(1 / np.sqrt(precp) - sigma_prot))
        perf['rna']['sigma_l1'] = np.mean(np.abs(1 / np.sqrt(precr) - sigma_rna))
        performance[n_subsamples]['mofa'] = perf

        # UNIMODAL
        # PathfA
        set_seed(seed)
        params, losses, neg_margliks = path_fa(
            Y_rna_sub, C_rna, n_latents=U.shape[1], positive_U=True,
            empirical_bayes=True, n_epochs=n_epochs_pathfa, svd_init=svd_init
        )
        perf_rna = compute_single_performances(Y_rna_test, C_rna, U, B_test, sigma_rna, params)
        perf_rna['nlj'] = losses[-1]
        perf_rna['nlml'] = neg_margliks[-1]
        params, losses, neg_margliks = path_fa(
            Y_prot_sub, C_prot, n_latents=U.shape[1], positive_U=True,
            empirical_bayes=True, n_epochs=n_epochs_pathfa, svd_init=svd_init
        )
        perf_prot = compute_single_performances(Y_prot_test, C_prot, U, B_test, sigma_prot, params)
        perf_prot['nlj'] = losses[-1]
        perf_prot['nlml'] = neg_margliks[-1]
        performance[n_subsamples]['pathfa'] = dict(rna=perf_rna, prot=perf_prot)

        # PLIER R
        set_seed(seed)
        with HiddenPrint():
            params = run_plier_R(Y_rna_sub, C_rna, n_latents=U.shape[1], n_iterations=n_epochs_plier)
        params['delta_B'] = np.ones(U.shape[1]) * params['delta_B']
        perf_rna = compute_single_performances(Y_rna_test, C_rna, U, B_test, sigma_rna, params)
        with HiddenPrint():
            params = run_plier_R(Y_prot_sub, C_prot, n_latents=U.shape[1], n_iterations=n_epochs_plier)
        params['delta_B'] = np.ones(U.shape[1]) * params['delta_B']
        perf_prot = compute_single_performances(Y_prot_test, C_prot, U, B_test, sigma_prot, params)
        performance[n_subsamples]['plier'] = dict(rna=perf_rna, prot=perf_prot)

        # FA
        fa = FactorAnalysis(n_components=U.shape[1], svd_method='lapack', random_state=seed)
        fa.fit(Y_rna_sub.T)
        nll_rna = log_likelihood(fa.components_.T @ fa.transform(Y_rna_test.T).T, C_rna, U, B_test, noise_prec=1/np.square(sigma_rna)).item() / np.prod(Y_rna_test.shape)
        sigma_l1_rna = np.mean(np.abs(np.sqrt(fa.noise_variance_) - sigma_rna.ravel()))
        fa.fit(Y_prot_sub.T)
        nll_prot = log_likelihood(fa.components_.T @ fa.transform(Y_prot_test.T).T, C_prot, U, B_test, noise_prec=1/np.square(sigma_prot)).item() / np.prod(Y_prot_test.shape)
        sigma_l1_prot = np.mean(np.abs(np.sqrt(fa.noise_variance_) - sigma_prot.ravel()))
        performance[n_subsamples]['fa'] = dict(
            rna=dict(nll=nll_rna, sigma_l1=sigma_l1_rna),
            prot=dict(nll=nll_prot, sigma_l1=sigma_l1_prot)
        )

        # Perf upper bound
        nll_rna = log_likelihood(C_rna @ U @ B_test, C_rna, U, B_test, noise_prec=1/np.square(sigma_rna)).item() / np.prod(Y_rna_test.shape)
        nll_prot = log_likelihood(C_prot @ U @ B_test, C_prot, U, B_test, noise_prec=1/np.square(sigma_prot)).item() / np.prod(Y_prot_test.shape)
        performance[n_subsamples]['optimum'] = dict(rna=nll_rna, prot=nll_prot)

    with open(f'synthetic_results/modes={n_modes}_bnoise={B_noise_scale}_seed={seed}_hetb={het_B}_hetnoise={het_noise}.pkl', 'wb') as f:
        pickle.dump(performance, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int)
    parser.add_argument('-m', '--modes', type=int, default=3)
    parser.add_argument('--B_noise_scale', type=float, default=1.0)
    parser.add_argument('--het_B', action='store_true')
    parser.add_argument('--no-het_B', dest='het_B', action='store_false')
    parser.add_argument('--het_noise', action='store_true')
    parser.add_argument('--no-het_noise', dest='het_noise', action='store_false')
    parser.set_defaults(het_B=False, het_noise=True)
    args = parser.parse_args()
    main(seed=args.seed, n_modes=args.modes, B_noise_scale=args.B_noise_scale, 
         het_noise=args.het_noise, het_B=args.het_B)
