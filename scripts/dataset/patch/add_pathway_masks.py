import anndata
from pathlib import Path
from absl import flags, app, logging

from pathfa import utils, ROOT

FLAGS = flags.FLAGS

MSIGDB_VERSION = 'v7.2'

flags.DEFINE_multi_enum(
        'name',
        'all',
        ['all', 'hallmark', 'oncogenic', 'immunologic', 'cell_type'],
        'Named version of pathway'
)

flags.DEFINE_boolean(
        'require_matched_ensg',
        False,
        'For e.g. proteomics, require a "matcehd_ensg" var field'
)


NICKNAME_TRANSLATION = {
        'oncogenic': 'msigdb/c6.oncogenic',
        'immunologic': 'msigdb/c7.immunologic',
        'cell_type': 'msigdb/c8.cell_type',
        'hallmark': 'msigdb/hallmark',
}

MAP_NAME_TO_PATH = {
        'msigdb/hallmark': f'msigdb/h.all.{MSIGDB_VERSION}.ensg.gmt',
        'msigdb/c6.oncogenic': f'msigdb/c6.all.{MSIGDB_VERSION}.ensg.gmt',
        'msigdb/c7.immunologic': f'msigdb/c7.all.{MSIGDB_VERSION}.ensg.gmt',
        'msigdb/c8.cell_type': f'msigdb/c8.all.{MSIGDB_VERSION}.ensg.gmt',
}


def main(argv):
    _, path = argv
    path = Path(path)
    data = anndata.read(path)

    if 'all' in FLAGS.name:
        keys = list(MAP_NAME_TO_PATH.keys())
    else:
        keys = FLAGS.name

    if FLAGS.require_matched_ensg:
        features_as_ensg = data.var['matched_ensg']
    else:
        features_as_ensg = None

    for key in keys:
        key = NICKNAME_TRANSLATION.get(key, key)
        membership_path = ROOT/'data'/'pathways'/MAP_NAME_TO_PATH[key]

        logging.info('Adding %s from %s', key, membership_path)
        membership = utils.load_pathway_membership(membership_path)
        mask = utils.make_mask(
                membership,
                data.var_names,
                features_as_ensg
        ).T

        outpath = path.parent/'pathways'/f'{key}.tsv.gz'
        outpath.parent.mkdir(exist_ok=True, parents=True)
        mask.to_csv(outpath, sep='\t')
        print(outpath)

    return


if __name__ == '__main__':
    app.run(main)
