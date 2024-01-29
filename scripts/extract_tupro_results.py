from itertools import product
import pandas as pd
from itertools import product
import numpy as np


RES_DIR = 'tupro_results/'


"""1. extract maximum correlating pathways with heterogneity and print latex table"""
def get_performance_uni(tumor, method, modality, geneset, sample_preproc='quantile_normalize', 
                        unit='rpm', z_score=True, latents=10, seed=None):
    if method == 'fa':
        geneset = 'all'
    run_name = f'{method}_{modality}_{geneset}_latents={latents}'
    run_name += '_' + unit + '_' + sample_preproc + '_log'
    if z_score:
        run_name += '_zscore'
    if seed is not None:
        run_name += f'_{seed}'
    U = pd.read_csv(RES_DIR + tumor + '/' + run_name + '/U.csv', index_col=0)
    perfs = pd.read_csv(RES_DIR + tumor + '/' + run_name + '/performance.csv', index_col=0).rename({'Fibroblasts': 'Fibroblast'}, axis=1)
    return perfs, U

def get_performance_multi(tumor, method, geneset, rna_sample_preproc='quantile_normalize', rna_unit='rpm', rna_zscore=True,
                          prot_sample_preproc='quantile_normalize', prot_unit='rpm', prot_zscore=True, latents=10, seed=None):
    if method == 'mofa':
        geneset = 'all'
    rna_flags = f'{rna_unit}_{rna_sample_preproc}' + ('_zscore' if rna_zscore else '')
    prot_flags = f'{prot_unit}_{prot_sample_preproc}' + ('_zscore' if prot_zscore else '')
    run_name = f'{method}_{geneset}_latents={latents}_{rna_flags}_{prot_flags}'
    if seed is not None:
        run_name += f'_{seed}'
    U = pd.read_csv(RES_DIR + tumor + '/' + run_name + '/U.csv', index_col=0)
    perfs = pd.read_csv(RES_DIR + tumor + '/' + run_name + '/performance.csv', index_col=0).rename({'Fibroblasts': 'Fibroblast'}, axis=1)
    return perfs, U

index = [
    np.array(['multi'] * 2),
    np.array(['pathway', 'correlation'])
]
df = pd.DataFrame(
    columns=[f'rank {i+1}' for i in range(10)], index=index
)
perf, U = get_performance_multi('ovarian', 'pathfa', 'msigdb-hallmark')
perf = perf.loc[U.index, 'jensenshannon'].sort_values(ascending=False)[:10]
df = pd.DataFrame(index=[f'rank {i+1}' for i in range(10)])
df['rank'] = range(1, 11, 1)
df['pathway'] = ['_'.join(e.split('_')[1:]) for e in perf.index]
df['correlation'] = perf.values
print(df.to_latex(index=False, float_format="{:0.2f}".format))

"""2. Extract correlations with mean and standard error over left out samples"""
def get_performance_uni(tumor, method, modality, geneset, sample_preproc='quantile_normalize', 
                        unit='rpm', z_score=True, latents=10, seed=0):
    if method == 'fa':
        geneset = 'all'
    run_name = f'{method}_{modality}_{geneset}_latents={latents}'
    run_name += '_' + unit + '_' + sample_preproc
    if z_score:
        run_name += '_zscore'
    if seed is not None:
        run_name += f'_{seed}'
    perfs = pd.read_csv(RES_DIR + tumor + '/' + run_name + '/performance.csv', index_col=0)
    return perfs.max().rename({'Fibroblasts': 'Fibroblast'})

def get_performance_multi(tumor, method, geneset, rna_sample_preproc='quantile_normalize', rna_unit='rpm', rna_zscore=True,
                          prot_sample_preproc='quantile_normalize', prot_unit='rpm', prot_zscore=True, latents=10, seed=0):
    rna_flags = f'{rna_unit}_{rna_sample_preproc}' + ('_zscore' if rna_zscore else '')
    prot_flags = f'{prot_unit}_{prot_sample_preproc}' + ('_zscore' if prot_zscore else '')
    if method == 'mofa':
        geneset = 'all'
    run_name = f'{method}_{geneset}_latents={latents}_{rna_flags}_{prot_flags}'
    if seed is not None:
        run_name += f'_{seed}'
    perfs = pd.read_csv(RES_DIR + tumor + '/' + run_name + '/performance.csv', index_col=0)
    return perfs.max().rename({'Fibroblasts': 'Fibroblast'})

tumors = ['ovarian', 'melanoma']
genesets = [
    'curated-melanoma-cell_type',
    'msigdb-c8.cell_type',
    'msigdb-hallmark',
    'all'
]
n_latent_variables = 10
seeds = [117, 711, 420, 523, 187, 982, 766, 233, 523, 832]
uni_methods = ['pathfa', 'fa', 'plier']
multi_methods = ['pathfa', 'mofa']

res_dict = dict()
for tumor, geneset in product(tumors, genesets):
    frame = pd.DataFrame(columns=['Tumor', 'Immune', 'Endothelial', 'Fibroblast'])
    if tumor == 'ovarian':
        frame = pd.DataFrame(columns=['Tumor', 'Immune', 'Endothelial', 'Fibroblast', 'jensenshannon'])
    for seed in seeds:
        if 'ovarian' in geneset and tumor != 'ovarian':
            continue
        if 'curated' in geneset and tumor != 'melanoma':
            continue

        # unimodal datasets
        for modality in ['rna', 'prot']:
            for method in uni_methods:
                name = '-'.join([method, modality])
                ixn = name + f'_{seed}'
                try:
                    frame.loc[ixn] = get_performance_uni(tumor, method, modality, geneset, 
                                                          latents=n_latent_variables, seed=seed)
                    frame.loc[ixn, ['name', 'modality', 'method', 'seed']] = [name, modality, method, seed]
                except Exception as e:
                    pass

        for method in multi_methods:
            name = '-'.join([method, 'multi'])
            ixn = name + f'_{seed}'
            try:
                frame.loc[ixn] = get_performance_multi(tumor, method, geneset, latents=n_latent_variables, seed=seed)
                frame.loc[ixn, ['name', 'modality', 'method', 'seed']] = [name, 'multi', method, seed]
            except Exception as e:
                pass
        res_dict[tumor + '-' + geneset] = frame

for key in res_dict:
    res_dict[key].to_csv(f'tupro_results/{key}_raw.csv')
    res_dict[key].groupby('name').mean().drop('seed', axis=1).to_csv(f'tupro_results/{key}_mean.csv')
    res_dict[key].groupby('name').sem().drop('seed', axis=1).to_csv(f'tupro_results/{key}_stderr.csv')
    
"""3. Extract pathway-cell-type correlation matrices for different genesets using multimodal PathFA"""
def get_performance_multi(tumor, method, geneset, rna_sample_preproc='quantile_normalize', rna_unit='rpm', rna_zscore=True,
                          prot_sample_preproc='quantile_normalize', prot_unit='rpm', prot_zscore=True, latents=10, seed=0):
    rna_flags = f'{rna_unit}_{rna_sample_preproc}' + ('_zscore' if rna_zscore else '')
    prot_flags = f'{prot_unit}_{prot_sample_preproc}' + ('_zscore' if prot_zscore else '')
    if method == 'mofa':
        geneset = 'all'
    run_name = f'{method}_{geneset}_latents={latents}_{rna_flags}_{prot_flags}'
    if seed is not None:
        run_name += f'_{seed}'
    perfs = pd.read_csv(RES_DIR + tumor + '/' + run_name + '/performance.csv', index_col=0)
    perfs = perfs.rename({'Fibroblasts': 'Fibroblast'}, axis=1).iloc[latents:]
    del perfs['ND']
    if 'jensenshannon' in perfs:
        del perfs['jensenshannon']
        del perfs['jensenshannon_log2']
    perfs['seed'] = seed
    return perfs

method = 'pathfa'
for tumor, geneset in product(tumors, genesets):
    if 'ovarian' in geneset and tumor != 'ovarian':
        continue
    if 'curated' in geneset and tumor != 'melanoma':
        continue
    if geneset == 'all':
        continue
    
    df = pd.concat([get_performance_multi(tumor, method, geneset, seed=seed) for seed in seeds])
    df.to_csv(f'tupro_results/{tumor}-{geneset}-corr_matrix.csv')