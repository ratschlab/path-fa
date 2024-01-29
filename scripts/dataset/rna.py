import anndata
import pandas as pd

from absl import flags, app
from pathlib import Path

FLAGS = flags.FLAGS
flags.DEFINE_string('outdir', None, 'Name of outdir')
flags.DEFINE_integer('patient_set_version', 1, 'set version')
flags.DEFINE_enum('patient_set', 'all',
                  ['all', 'melanoma'],
                  'Name of patient set')

flags.mark_flag_as_required('outdir')


def load_rna(tsv):
    df = pd.read_csv(tsv, sep='\t', index_col=0).T
    return anndata.AnnData(df)


def main(argv):
    _, path = argv
    path = Path(path)

    data = load_rna(path)
    data.var_names = data.var_names.str.replace(r'\..*$', '')
    _, _, feature_key, preproc_keys, _ = path.stem.split('-', 4)

    if FLAGS.patient_set == 'melanoma':
        mask = data.obs_names.str.startswith('M')
        data = data[mask].copy()

    patient_key = f'{FLAGS.patient_set}_v{FLAGS.patient_set_version}'

    data.uns['patient_set'] = patient_key
    data.uns['feature_set'] = feature_key
    data.uns['preproc'] = preproc_keys

    outdir = Path(FLAGS.outdir)/f'{patient_key}-{feature_key}'
    outdir.mkdir(parents=True, exist_ok=True)
    data.write(outdir/f'{preproc_keys}.h5ad')
    return


if __name__ == '__main__':
    app.run(main)
