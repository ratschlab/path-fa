import anndata
import numpy as np
import scanpy as sc

from absl import flags, app
from pathlib import Path
from toolz.functoolz import pipe


FLAGS = flags.FLAGS

flags.DEFINE_string('outdir', None, 'Name of outdir')
flags.DEFINE_string('patient_set', None, 'Name of patient set')
flags.DEFINE_float('missing_feature_thold', 0.25,
                   'Filter features missing more than X pct',
                   lower_bound=0, upper_bound=1)
flags.DEFINE_boolean('library_size_norm', True, 'Normalize library size')
flags.DEFINE_boolean('log1p', True, 'Apply log(1 + X)')
flags.DEFINE_boolean('debug', False, 'Produces debugging output.')
flags.mark_flag_as_required('outdir')
flags.mark_flag_as_required('patient_set')


def libnorm(data, quantile=85, axis=1, **kwargs):
    factor = np.nanpercentile(data, quantile, axis=axis, **kwargs)
    return data / factor[:, np.newaxis]


def main(argv):
    _, *paths = argv

    data = pipe(
        paths,
        lambda x: map(anndata.read, x),
        lambda data: anndata.concat(data, axis=0, join='inner', merge='same')
    )

    feature_key = f'filter_missing_0{int(100*FLAGS.missing_feature_thold)}'
    feature_mask = np.isnan(data.X).mean(0) < FLAGS.missing_feature_thold
    data = data[:, feature_mask].copy()
    data.layers['raw'] = data.X.copy()

    preproc_keys = ['zero_impute']
    data.X[np.isnan(data.X)] = 0

    if FLAGS.library_size_norm:
        preproc_keys.append('library_size_norm')
        data.X = libnorm(data.X)

    if FLAGS.log1p:
        preproc_keys.append('log1p')
        sc.pp.log1p(data)

    preproc_keys = preproc_keys[::-1]

    data.uns['patient_set'] = FLAGS.patient_set
    data.uns['feature_set'] = feature_key
    data.uns['preproc'] = preproc_keys

    preproc_keys = preproc_keys.replace('_', '-')

    outdir = Path(FLAGS.outdir)/f'{FLAGS.patient_set}-{feature_key}'
    outdir.mkdir(parents=True, exist_ok=True)
    data.write(outdir/f'{preproc_keys}.h5ad')
    return


if __name__ == '__main__':
    app.run(main)
