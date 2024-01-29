import numpy as np
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from rpy2.robjects import NULL
plier_R = importr('PLIER')


def plier(Y, C, n_latents, n_iterations=350, eps=1e-6, scale=False):
    C = C.astype(float)
    rpy2.robjects.numpy2ri.activate()
    Yr = ro.r.matrix(Y, nrow=Y.shape[0], ncol=Y.shape[1])
    ro.r.assign('Y', Yr)
    Cr = ro.r.matrix(C, nrow=C.shape[0], ncol=C.shape[1])
    ro.r.assign('C', Cr)
    rpy2.robjects.numpy2ri.deactivate()
    n_latents = NULL if n_latents == None else n_latents
    res = plier_R.PLIER(Yr, Cr, doCrossval=False, k=n_latents, scale=scale, 
                        max_iter=n_iterations, tol=eps)
    params = {'U': np.array(res[3]), 'Z': np.array(res[2]), 
              'B': np.array(res[1]), 'delta_B': res[6]}
    return params
    

if __name__ == '__main__':
    n_genes = 5000
    n_pathways = 50
    n_samples = 30
    M, Ci = np.random.randn(n_genes, n_samples), np.random.randint(0, 2, (n_genes, n_pathways))
    plier(M, Ci, 5)
