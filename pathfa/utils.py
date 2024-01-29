import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.preprocessing import quantile_transform
import logging

import anndata


TUPRO_PATH = '/cluster/work/tumorp/share/pathfa'

# samples that have prot, rna, cytof, and clinical information available
subset_melanoma_samples = [
    'macegej',
    'madegod',
    'mahobam',
    'majefev',
    'malylej',
    'mapoxub',
    'mecuguh',
    'mecygyr',
    'meducin',
    'medycar',
    'mehufef',
    'mehylob',
    'mekakyd',
    'mekobab',
    'melipit',
    'mifimab',
    'migofiw',
    'mihifib',
    'milabyl',
    'mimanar',
    'mimuvyf',
    'mipynap',
    'modudol',
    'mogylip',
    'mololyb',
    'motamuh',
    'mucadop',
    'mufacik',
    'mugakol',
    'mukagox',
    'mumifad',
    'myjilas',
    'mykokig',
    'mynelic',
]
subset_melanoma_samples = list(map(lambda s: s.upper(), subset_melanoma_samples))

# intersecting samples rna, prot, cytof composition
subset_ovarian_samples = [
    'OKW5U',
    'OB225',
    'ODAJEGA',
    'OCABUGY',
    'ODIDUKU',
    'OQADOQU',
    'OKAGOCY',
    'OTEWUZO',
    'OHACAHY',
    'OHAMUME',
    'OQEMOGA',
    'OHANIQY',
    'OHAHEBY',
    'OVAMUZI',
    'OHEJAFA',
    'OPAPUHU',
    'ADEPALU',
    'OXEXYCY',
    'OMABOPY',
    'ABABECE',
    'ODEGOHY',
    'ODAQUPA',
    'ORAVAFU',
    'ORAJIMY',
    'OLEFUTI',
    'OMAJORO',
    'AWITEXO',
    'OKEBYDA',
    'ADEGAKO',
    'AFACYKY',
    'OQADIRY',
    'OQEFIRU',
    'AWOVAXI',
    'OLEKESU',
    'OZUVAPU',
    'OPELICI',
    'OCANYTE',
    'OMAJIQI',
    'OMEFOMO',
    'OXUMUZU',
    'OKAHEDU',
    'OMAFYSU'
]


def load_pathway_membership(path):
    membership = dict()
    for line in open(path, 'r'):
        key, _, *ensgs = line.strip().split('\t')
        membership[key] = ensgs
    return membership


def make_mask(membership, features, features_as_ensg=None):
    '''Creates a pathway-by-feature mask representing membership

    The mask is created via a CSR matrix.
    This is much faster for large pathway sets,
    where iterative updates to a dense matrix becomes very expensive
    '''
    if features_as_ensg is None:
        features_as_ensg = features

    def iter_ensg_membership(membership, features_as_ensg):
        '''Match the ensg in pathway to the index in feature set
        '''
        for key, ensg in tqdm(membership.items()):
            featmask = np.in1d(features_as_ensg, ensg)
            cidx, = np.where(featmask)
            yield cidx

    shape = (len(membership), len(features))
    csr = csr_from_indices(
            iter_ensg_membership(membership, features_as_ensg),
            shape=shape,
            dtype=bool)

    mask = pd.DataFrame(
        np.array(csr.todense()),
        index=membership.keys(),
        columns=features)

    assert mask.values.sum() > 0

    return mask


def csr_from_indices(iter_cidx, **kwargs):
    indices = list()
    indptr = [0]
    for cidx in iter_cidx:
        indices.extend(list(cidx))
        indptr.append(indptr[-1] + len(cidx))

    csr = sparse.csr_matrix(
        (np.ones_like(indices), indices, indptr),
        **kwargs
    )

    return csr


def get_genesets():
    rna = anndata.read(f'{TUPRO_PATH}/melanoma/rna/normalized_sizefactor_zscore.h5ad')
    return list(rna.varm.keys())


def get_data_preprocessed(dataset, geneset, tumor='melanoma', unit='rpm', filter_genes=False, log=True, 
                          sample_preprocessing='quantile_normalize', z_score=False, subset=False):
    assert tumor in ['melanoma', 'ovarian']
    assert unit in ['rpkm', 'rpm', 'raw']
    assert sample_preprocessing in ['quantile_normalize', 'standardize', 'none']
    
    # start from raw counts in data.layers['counts']
    if dataset == 'rna':
        assert unit in ['rpkm', 'rpm', 'raw']
        rna = anndata.read(f'{TUPRO_PATH}/{tumor}/rna/normalized_sizefactor_zscore.h5ad')
        all_ensg_ids = rna.to_df().columns
        if tumor == 'melanoma':
            if subset == 'intersection':
                rna = rna[subset_melanoma_samples]
            elif subset is not None and 'intersection' in subset:
                seed = int(subset.split('_')[1])
                np.random.seed(seed)
                subset_samples = np.random.permutation(subset_melanoma_samples)[:-4]
                rna = rna[subset_samples]
            else:
                assert not subset
            count_data = rna.layers['counts']  # samples x markers
            rna_df = rna.to_df()
        else:
            # retrieve raw count data for ovarian
            count_data = pd.read_csv(f'{TUPRO_PATH}/ovarian/rna/TuPro-ovca-pcgenes-rawcounts-v3.tsv', index_col=0, sep='\t').T
            if subset == 'intersection':
                count_data = count_data.loc[subset_ovarian_samples]
            elif subset is not None and 'intersection' in subset:
                seed = int(subset.split('_')[1])
                np.random.seed(seed)
                subset_samples = np.random.permutation(subset_ovarian_samples)[:-4]
                count_data = count_data.loc[subset_samples]
            else:
                assert not subset
                count_data = count_data.loc[count_data.index.isin(set(qc_ovarian_samples))]
            count_data.columns = count_data.columns.map(lambda s: s.split('.')[0])
            count_data = count_data[all_ensg_ids]
            count_data = count_data.groupby(axis=1, level=0).sum()
            rna_df = count_data.copy()
            count_data = count_data.values

        # Don't need to filter genes with zero counts as suggested in PLIER paper since there are none here.
        # 1. RPKM
        if unit in ['rpkm', 'rpm']:
            # Retrieve gene_lengths
            gls = pd.read_csv(f'{TUPRO_PATH}/gencode.v32.gene_lengths.tsv', sep='\t', index_col=0)
            # remove nr in `EMSMBL.nr`
            gls.index = gls.index.map(lambda x: x.split('.')[0])
            # there are just duplicates but they consistently have the same gene_length fields so safe to discared without aggregation
            gls = gls[~gls.index.duplicated(keep='first')]
            # subset to variables in the data
            gene_lengths = gls.loc[rna.var_names, 'gene_length'].values.reshape(1, -1)

            # Compute library size
            library_size = count_data.sum(1).reshape(-1, 1)

            if unit == 'rpkm':
                data = count_data * (10 ** 9) / library_size / gene_lengths
            else:  # unit = 'rpm'
                average_library_size = library_size.mean()
                data = count_data * average_library_size / library_size

        else:  # unit == 'raw'
            data = count_data

        # 2. filter genes that have < 3 counts for any sample (also prevents zero-inflation)
        if filter_genes:
            mask = ~np.any(count_data < 3, axis=0)
            data = data[:, mask]

        # 3. log data
        if log and filter_genes:  # min count is 3 so can safely apply log
            data = np.log(data)
        else:  # min count could be 0
            data = np.log(1 + data)

        # 3. preprocess samples (quantile_norm or zscore/standardization or nothing)
        if sample_preprocessing == 'quantile_normalize':
            # transform to normal over markers per sample
            data = quantile_transform(data.T, output_distribution='normal').T 
        elif sample_preprocessing == 'standardize':
            # divide mean and std per sample
            m, s = data.mean(axis=1), data.std(axis=1)
            data = (data - m.reshape(-1, 1)) / s.reshape(-1, 1)
        elif log:
            # at least set to zero mean (unless we don't log and keep count data)
            data = data - data.mean()

        # 4. z_score (marker preprocessing)
        if z_score:
            m, s = data.mean(axis=0), data.std(axis=0)
            data = (data - m.reshape(1, -1)) / s.reshape(1, -1)

        if filter_genes:
            Y_obs = pd.DataFrame(data, index=rna_df.index, columns=rna_df.columns[mask]).T
        else:
            Y_obs = pd.DataFrame(data, index=rna_df.index, columns=rna_df.columns).T
            
        if geneset is None:
            return Y_obs, None
        # retrieve mask and filter based on mask on markers
        if filter_genes:
            C_mask = rna.varm[geneset].loc[rna_df.columns[mask]]
        else:
            C_mask = rna.varm[geneset].loc[rna_df.columns]

        joint_ixs = C_mask.index.intersection(Y_obs.index)
        C_mask = C_mask.loc[joint_ixs]
        Y_obs = Y_obs.loc[joint_ixs]

    elif dataset == 'prot':
        assert unit in ['raw', 'rpm']
        # load proteomics mask data
        prot_mask = anndata.read(f'{TUPRO_PATH}/melanoma/prot/zscore.h5ad')

        # load "raw" data
        if tumor == 'melanoma':
            Y_obs = pd.read_csv(f'{TUPRO_PATH}//melanoma/prot/20210107_154540_TP_Melanoma_January_2021_Peptides_QValue_NoFilter_Report_batch-adjusted_protein.csv', index_col=0)
        elif tumor == 'ovarian':
            Y_obs = pd.read_csv(f'{TUPRO_PATH}/ovarian/prot/20210413_092626_14052020_OVCA_all_April_2021_Peptides_QValue_NoFilter_Report_batch-adjusted_protein_v2.csv', index_col=0)
        else:
            raise ValueError()

        Y_obs.columns = [e.split('_')[0] for e in Y_obs.columns]
        if subset == 'intersection':
            if tumor == 'melanoma':
                Y_obs = Y_obs[subset_melanoma_samples]
            else:
                Y_obs = Y_obs[subset_ovarian_samples]
        elif subset is not None and 'intersection' in subset:
            seed = int(subset.split('_')[1])
            np.random.seed(seed)
            if tumor == 'melanoma':
                subset_samples = np.random.permutation(subset_melanoma_samples)[:-4]
            else:
                subset_samples = np.random.permutation(subset_ovarian_samples)[:-4]
            Y_obs = Y_obs[subset_samples]
        else:
            assert not subset
            if tumor == 'ovarian':
                Y_obs = Y_obs[Y_obs.columns.intersection(qc_ovarian_samples)]
        data = Y_obs.T.values

        # 1. pseudo rpm (normalize by "library size")
        if unit == 'rpm':
            library_size = data.sum(axis=1).reshape(-1, 1)
            average_library_size = library_size.mean()
            data = data / library_size * average_library_size

        # 2. preprocess samples (quantile_norm or zscore/standardization or nothing)
        if sample_preprocessing == 'quantile_normalize':
            # transform to normal over markers per sample
            data = quantile_transform(data.T, output_distribution='normal').T 
        elif sample_preprocessing == 'standardize':
            # divide mean and std per sample
            m, s = data.mean(axis=1), data.std(axis=1)
            data = (data - m.reshape(-1, 1)) / s.reshape(-1, 1)
        elif log:
            # at least set to zero mean (unless we don't log and keep count data)
            data = data - data.mean()
        
        # 3. marker preprocessing (zscore is standard but nothing is required)
        if z_score:
            m, s = data.mean(axis=0), data.std(axis=0)
            data = (data - m.reshape(1, -1)) / s.reshape(1, -1)

        Y_obs.iloc[:, :] = data.T
        if geneset is None:
            return Y_obs, None
        C_mask = prot_mask.varm[geneset]
        joint_ixs = C_mask.index.intersection(Y_obs.index)
        C_mask = C_mask.loc[joint_ixs]
        Y_obs = Y_obs.loc[joint_ixs]

    else:
        raise ValueError('Invalid dataset:', dataset)

    # keep only activated markers
    marker_in_mask = (C_mask.sum(axis=1) != 0).values
    logging.info(f'Keeping {marker_in_mask.sum()} markers ({100*marker_in_mask.sum()/len(marker_in_mask):.2f}%)')
    return Y_obs[marker_in_mask], C_mask[marker_in_mask]


def get_multimodal_data_preprocessed(geneset, tumor, rna_config, prot_config):
    Y_rna, C_rna = get_data_preprocessed('rna', geneset, tumor, **rna_config)
    Y_prot, C_prot = get_data_preprocessed('prot', geneset, tumor, **prot_config)
    logging.info(f'Patients with RNA data: {Y_rna.shape[1]}; with Prot data: {Y_prot.shape[1]}')
    joint_patients = Y_prot.columns.intersection(Y_rna.columns)
    logging.info(f'Number of joint patients: {len(joint_patients)}')
    Y_rna, Y_prot = Y_rna[joint_patients], Y_prot[joint_patients]
    return Y_rna, C_rna, Y_prot, C_prot

    
def compute_correlation_performance(representation, composition):
    # assume representation is patient x latent and composition is patient x types
    df_perf = pd.DataFrame(index=representation.columns, columns=composition.columns)
    joint_ixs = representation.index.intersection(composition.index)
    for rep in representation.columns:
        for ctype in composition.columns:
            df_perf.loc[rep, ctype] = pearsonr(representation.loc[joint_ixs, rep].values, 
                                               composition.loc[joint_ixs, ctype].values)[0]
    return df_perf
