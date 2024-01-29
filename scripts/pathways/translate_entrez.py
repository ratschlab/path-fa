import re
import sys
import mygene


def parse(path):
    store = dict()
    uniq_entrez = set()
    for line in open(path, 'r'):
        key, source, *entrez = line.strip().split('\t')
        uniq_entrez.update(entrez)
        store[(key, source)] = entrez

    return uniq_entrez, store


def query(entrez):
    '''Query all ensg IDs from entrez IDs
    '''
    map_entrez_to_ensg = dict()
    mg = mygene.MyGeneInfo()
    ginfo = mg.querymany(entrez, scopes='entrezgene', fields='ensembl.gene')
    for gene in ginfo:
        entrez = gene['query']
        try:
            res = gene['ensembl']
        except (TypeError, KeyError):
            print('Bad query', gene)
            continue

        if isinstance(res, list):
            ensg = res[0]['gene']
        else:
            ensg = res['gene']
        map_entrez_to_ensg[entrez] = ensg

    return map_entrez_to_ensg


if __name__ == '__main__':
    _, path = sys.argv
    outpath = re.sub('entrez.gmt$', 'ensg.gmt', path)

    uniq_entrez, store = parse(path)
    map_entrez_to_ensg = query(uniq_entrez)

    with open(outpath, 'w') as fout:
        for (key, source), entrez in store.items():
            ensg = [map_entrez_to_ensg[x] for x in entrez if x in map_entrez_to_ensg]
            fout.write('\t'.join([key, source] + ensg) + '\n')
