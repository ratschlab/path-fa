import numpy as np
import torch
from pathlib import Path
from absl import app
from absl import flags
import pandas as pd

from mofapy.core.entry_point import entry_point

from pathfa.path_fa import multi_path_fa
from pathfa.utils import TUPRO_PATH, get_genesets, get_multimodal_data_preprocessed, compute_correlation_performance


FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 711, 'Random seed for data generation and model initialization.')
flags.DEFINE_enum('geneset', None, get_genesets() + ['all'], 'Geneset/Pathwayset to work with.', required=True)
flags.DEFINE_enum('tumor', 'melanoma', ['melanoma', 'ovarian'], 'Tumor type.')
flags.DEFINE_integer('n_latents', 10, 'Number of latent factors.', lower_bound=1)
flags.DEFINE_integer('n_epochs', 100, 'Number of training epochs', lower_bound=0)
flags.DEFINE_float('lr', 1e-1, 'Learning rate.')
flags.DEFINE_enum('device', 'cpu', ['cpu', 'cuda'], 'Device to run pathFA on.')
flags.DEFINE_bool('double', True, 'Whether to use double precision.')
flags.DEFINE_enum('method', 'pathfa', ['pathfa', 'mofa'], 'Method')
flags.DEFINE_string('result_dir', 'tupro_results/', 'Directory where to store results.')
flags.DEFINE_string('subset', 'False', 'subset of samples (False, intersection, intersection_seed)')

flags.DEFINE_enum('rna_unit', 'rpm', ['rpkm', 'rpm', 'raw'], 'data unit, raw are counts')
flags.DEFINE_enum('prot_unit', 'rpm', ['rpm', 'raw'], 'data unit, raw are counts')
flags.DEFINE_enum('rna_sample_preprocessing', 'quantile_normalize', ['quantile_normalize', 'standardize', 'none'], 
                  'preprocessing applied to samples')
flags.DEFINE_enum('prot_sample_preprocessing', 'quantile_normalize', ['quantile_normalize', 'standardize', 'none'], 
                  'preprocessing applied to samples')
flags.DEFINE_bool('rna_z_score', True, 'z-score markers (preproc applied to markers/features)')
flags.DEFINE_bool('prot_z_score', True, 'z-score markers (preproc applied to markers/features)')


def main(_):
    torch.manual_seed(FLAGS.seed)
    if FLAGS.geneset == 'all':
        geneset = None
    else:
        geneset = FLAGS.geneset
    subset = FLAGS.subset if FLAGS.subset != 'False' else False
    rna_config = dict(unit=FLAGS.rna_unit, sample_preprocessing=FLAGS.rna_sample_preprocessing,
                      z_score=FLAGS.rna_z_score, filter_genes=False, log=True, subset=subset)
    prot_config = dict(unit=FLAGS.prot_unit, sample_preprocessing=FLAGS.prot_sample_preprocessing,
                       z_score=FLAGS.prot_z_score, subset=subset)
    Y_rna, C_rna, Y_prot, C_prot = get_multimodal_data_preprocessed(geneset, FLAGS.tumor, rna_config, prot_config)

    # cytof cell type composition and heterogeneity for ovarian cancer samples
    if FLAGS.tumor == 'melanoma':
        composition_cytof = pd.read_csv(f'{TUPRO_PATH}/melanoma-composition-cytof-v1.tsv',
                                        sep='\t', index_col=0, usecols=[1, 2, 3, 4, 5, 6])
    else:
        composition_cytof = pd.read_csv(f'{TUPRO_PATH}/ovca-composition-cytof-v2.tsv', 
                                        sep='\t', index_col=0)
        heterogeneity = pd.read_csv(f'{TUPRO_PATH}/ovca-primary_neo_relapse-cytof-tumor-combined_clustering-all_measures.tsv', 
                                    sep='\t', index_col=1)
        composition_cytof = composition_cytof.merge(heterogeneity[['jensenshannon', 'jensenshannon_log2']], left_index=True, right_index=True)

    # RNA flags
    postfix = '_' + FLAGS.rna_unit + '_' + FLAGS.rna_sample_preprocessing
    if FLAGS.rna_z_score:
        postfix += '_zscore'
    # Prot flags
    postfix += '_' + FLAGS.prot_unit + '_' + FLAGS.prot_sample_preprocessing
    if FLAGS.prot_z_score:
        postfix += '_zscore'
    if subset and 'intersection' in subset and len(subset.split('_')) == 2:
        seed = subset.split('_')[1]
        postfix += f'_{seed}'
    run_name = f'{FLAGS.method}_{FLAGS.geneset}_latents={FLAGS.n_latents}{postfix}'

    results = Path(FLAGS.result_dir) / FLAGS.tumor
    if FLAGS.method == 'pathfa':
        assert FLAGS.geneset is not None
        params, losses, negmargliks = multi_path_fa(
            Y_rna.values, C_rna.values, Y_prot.values, C_prot.values, FLAGS.n_latents, 
            n_epochs=FLAGS.n_epochs, lr=FLAGS.lr, positive_U=True, device=FLAGS.device, 
            double=FLAGS.double
        )
        U, B = params['U'], params['B']
        latents = list(range(1, FLAGS.n_latents+1))
        genesets = C_rna.columns
        patients = Y_rna.columns
        B = pd.DataFrame(B, index=latents, columns=patients)
        U = pd.DataFrame(U, index=genesets, columns=latents)
        cp = pd.DataFrame([[params['cp']]], index=['all'], columns=['cp'])
        cr = pd.DataFrame([[params['cr']]], index=['all'], columns=['cr'])
        losses = pd.DataFrame([losses, negmargliks], index=['losses', 'neg_margliks']).T
        perfs = pd.concat([compute_correlation_performance(B.T, composition_cytof),
                           compute_correlation_performance((U @ B).T, composition_cytof)])
        sigma_noise_r = pd.DataFrame(params['sigma_noise_r'], index=Y_rna.index, columns=['all'])
        sigma_noise_p = pd.DataFrame(params['sigma_noise_p'], index=Y_prot.index, columns=['all'])
        delta_U = pd.DataFrame(params['delta_U'], index=genesets, columns=latents)
        delta_B = pd.DataFrame(params['delta_B'], index=latents, columns=['all'])
        p = results / run_name
        p.mkdir(exist_ok=True)
        Y_rna.to_csv(p / 'Y_rna.csv')
        C_rna.to_csv(p / 'C_rna.csv')
        Y_prot.to_csv(p / 'Y_prot.csv')
        C_prot.to_csv(p / 'C_prot.csv')
        B.to_csv(p / 'B.csv')
        U.to_csv(p / 'U.csv')
        sigma_noise_r.to_csv(p / 'sigma_noise_r.csv')
        sigma_noise_p.to_csv(p / 'sigma_noise_p.csv')
        delta_B.to_csv(p / 'delta_B.csv')
        delta_U.to_csv(p / 'delta_U.csv')
        losses.to_csv(p / 'losses.csv', index=False)
        perfs.to_csv(p / 'performance.csv')
        cp.to_csv(p / 'cp.csv')
        cr.to_csv(p / 'cr.csv')
    
    elif FLAGS.method == 'mofa':
        ep = entry_point()
        Y_rna = Y_rna.iloc[(Y_rna.values.std(axis=1) != 0)]
        Y_prot = Y_prot.iloc[(Y_prot.values.std(axis=1) != 0)]
        data = [Y_rna, Y_prot]
        ep.set_data(data)
        ep.set_model_options(factors=FLAGS.n_latents, likelihoods=['gaussian', 'gaussian'], sparsity=True)
        ep.set_data_options(view_names=['rna', 'prot'], RemoveIncompleteSamples=False)
        ep.parse_data()
        ep.set_train_options(iter=FLAGS.n_epochs, tolerance=0.01, dropR2=0, elbofreq=1, 
                             verbose=False, seed=FLAGS.seed)
        ep.define_priors()
        ep.define_init()
        ep.parse_intercept()
        ep.train_model()
        p = results / run_name
        p.mkdir(exist_ok=True)
        expectations = ep.model.getExpectations(only_first_moments=True)
        latents = [str(i) for i in range(FLAGS.n_latents)]
        Z = pd.DataFrame(expectations['Z'], columns=latents, index=Y_prot.columns)
        perfs = compute_correlation_performance(Z, composition_cytof)
        W_rna = pd.DataFrame(expectations['SW'][0], index=Y_rna.index, columns=latents)
        W_prot = pd.DataFrame(expectations['SW'][1], index=Y_prot.index, columns=latents)
        S_rna = pd.DataFrame(1/expectations['Tau'][0], index=Y_rna.columns, columns=Y_rna.index)
        S_prot = pd.DataFrame(1/expectations['Tau'][1], index=Y_prot.columns, columns=Y_prot.index)
        Z.to_csv(p / 'Z.csv') 
        W_rna.to_csv(p / 'W_rna.csv')
        W_prot.to_csv(p / 'W_prot.csv')
        S_rna.to_csv(p / 'S_rna.csv')
        S_prot.to_csv(p / 'S_prot.csv')
        Y_rna.to_csv(p / 'Y_rna.csv')
        Y_prot.to_csv(p / 'Y_prot.csv')
        perfs.to_csv(p / 'performance.csv')


if __name__ == '__main__':
    app.run(main)
