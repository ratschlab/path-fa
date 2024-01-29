import numpy as np
import torch
import logging
from pathlib import Path
from absl import app
from absl import flags
import pandas as pd
from sklearn.decomposition import FactorAnalysis

from pathfa.path_fa import path_fa
from pathfa.plier_R import plier
from pathfa.utils import get_genesets, get_data_preprocessed, compute_correlation_performance, TUPRO_PATH


FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 711, 'Random seed for data generation and model initialization.')
flags.DEFINE_enum('dataset', 'rna', ['rna', 'prot'], 'Data modality to work with.')
flags.DEFINE_enum('geneset', None, get_genesets() + ['all'], 'Geneset/Pathwayset to work with.', required=True)
flags.DEFINE_enum('tumor', 'melanoma', ['melanoma', 'ovarian'], 'Tumor type.')
flags.DEFINE_enum('method', 'pathfa', ['pathfa', 'fa', 'plier'], 'Method')
flags.DEFINE_integer('n_latents', 10, 'Number of latent factors.', lower_bound=1)
flags.DEFINE_integer('n_epochs', 100, 'Number of training epochs', lower_bound=0)
flags.DEFINE_float('lr', 1e-1, 'Learning rate.')
flags.DEFINE_enum('device', 'cpu', ['cpu', 'cuda'], 'Device to run PathFA on.')
flags.DEFINE_bool('double', True, 'Whether to use double precision.')
flags.DEFINE_string('result_dir', 'tupro_results/', 'Directory where to store results.')
flags.DEFINE_string('subset', 'False', 'subset of samples (False, intersection, intersection_seed)')

flags.DEFINE_enum('unit', 'rpm', ['rpkm', 'rpm', 'raw'], 'data unit, raw are counts')
flags.DEFINE_enum('sample_preprocessing', 'quantile_normalize', ['quantile_normalize', 'standardize', 'none'], 
                  'preprocessing applied to samples')
flags.DEFINE_bool('z_score', True, 'z-score markers (preproc applied to markers/features)')


def main(_):
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    if FLAGS.geneset == 'all':
        geneset = None
    else:
        geneset = FLAGS.geneset
    subset = FLAGS.subset if FLAGS.subset != 'False' else False
    Y_obs, C_mask = get_data_preprocessed(
        FLAGS.dataset, geneset, FLAGS.tumor, FLAGS.unit, sample_preprocessing=FLAGS.sample_preprocessing,
        z_score=FLAGS.z_score, subset=subset
    )
    logging.info(f'Loaded {FLAGS.tumor} data. Shape of observations: {Y_obs.shape}.')

    if FLAGS.tumor == 'melanoma':
        composition_cytof = pd.read_csv(f'{TUPRO_PATH}/melanoma-composition-cytof-v1.tsv',
                                        sep='\t', index_col=0, usecols=[1, 2, 3, 4, 5, 6])
    else:
        composition_cytof = pd.read_csv(f'{TUPRO_PATH}/ovca-composition-cytof-v2.tsv', 
                                        sep='\t', index_col=0)
        heterogeneity = pd.read_csv(f'{TUPRO_PATH}/ovca-primary_neo_relapse-cytof-tumor-combined_clustering-all_measures.tsv', 
                                    sep='\t', index_col=1)
        composition_cytof = composition_cytof.merge(heterogeneity[['jensenshannon', 'jensenshannon_log2']], left_index=True, right_index=True)

    postfix = '_' + FLAGS.unit + '_' + FLAGS.sample_preprocessing
    if FLAGS.z_score:
        postfix += '_zscore'
    if subset and 'intersection' in subset and len(subset.split('_')) == 2:
        seed = subset.split('_')[1]
        postfix += f'_{seed}'
    run_name = f'{FLAGS.method}_{FLAGS.dataset}_{FLAGS.geneset}_latents={FLAGS.n_latents}{postfix}'

    results = Path(FLAGS.result_dir) / FLAGS.tumor

    if FLAGS.method == 'fa':
        fa = FactorAnalysis(n_components=FLAGS.n_latents, svd_method='lapack', random_state=FLAGS.seed)
        B = fa.fit_transform(Y_obs.values.T).T
        M = fa.components_.T
        latents = list(range(1, FLAGS.n_latents+1))
        B = pd.DataFrame(B, index=latents, columns=Y_obs.columns)
        M = pd.DataFrame(M, index=Y_obs.index, columns=latents)
        perfs = compute_correlation_performance(B.T, composition_cytof)
        p = results / run_name
        p.mkdir(exist_ok=True)
        B.to_csv(p / 'B.csv')
        M.to_csv(p / 'M.csv')
        perfs.to_csv(p / 'performance.csv')

    elif FLAGS.method == 'plier':
        params = plier(Y_obs.values, C_mask.values, FLAGS.n_latents,
                           n_iterations=FLAGS.n_epochs)
        Z, U, B = params['Z'], params['U'], params['B']
        latents = list(range(1, FLAGS.n_latents+1))
        Z = pd.DataFrame(Z, index=C_mask.index, columns=latents)
        U = pd.DataFrame(U, index=C_mask.columns, columns=latents)
        B = pd.DataFrame(B, index=latents, columns=Y_obs.columns)
        perfs = pd.concat([compute_correlation_performance(B.T, composition_cytof),
                           compute_correlation_performance((U @ B).T, composition_cytof)])
        p = results / run_name
        p.mkdir(exist_ok=True)
        Y_obs.to_csv(p / 'Y_obs.csv')
        C_mask.to_csv(p / 'C_mask.csv')
        B.to_csv(p / 'B.csv')
        Z.to_csv(p / 'Z.csv')
        U.to_csv(p / 'U.csv')
        perfs.to_csv(p / 'performance.csv')

    elif FLAGS.method == 'pathfa':
        assert FLAGS.geneset is not None
        params, losses, negmargliks = path_fa(
            Y_obs.values, C_mask.values, FLAGS.n_latents, n_epochs=FLAGS.n_epochs,
            lr=FLAGS.lr, device=FLAGS.device, double=FLAGS.double
        )
        U, B = params['U'], params['B']
        latents = list(range(1, FLAGS.n_latents+1))
        sigma_noise = pd.DataFrame(params['sigma_noise'], index=Y_obs.index, columns=['all'])
        U = pd.DataFrame(U, index=C_mask.columns, columns=latents)
        B = pd.DataFrame(B, index=latents, columns=Y_obs.columns)
        delta_U = pd.DataFrame(params['delta_U'], index=C_mask.columns, columns=latents)
        delta_B = pd.DataFrame(params['delta_B'], index=latents, columns=['all'])
        losses = pd.DataFrame([losses, negmargliks], index=['losses', 'neg_margliks']).T
        perfs = pd.concat([compute_correlation_performance(B.T, composition_cytof),
                           compute_correlation_performance((U @ B).T, composition_cytof)])
        p = results / run_name
        p.mkdir(exist_ok=True)
        Y_obs.to_csv(p / 'Y_obs.csv')
        C_mask.to_csv(p / 'C_mask.csv')
        sigma_noise.to_csv(p / 'sigma_noise.csv')
        B.to_csv(p / 'B.csv')
        U.to_csv(p / 'U.csv')
        delta_B.to_csv(p / 'delta_B.csv')
        delta_U.to_csv(p / 'delta_U.csv')
        losses.to_csv(p / 'losses.csv', index=False)
        perfs.to_csv(p / 'performance.csv')


if __name__ == '__main__':
    app.run(main)
