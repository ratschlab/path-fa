import io
import sys
import anndata
import pandas as pd
import urllib.parse
import urllib.request


def translate_via_uniprot(query, source='ACC+ID', target='ENSEMBL_ID'):

    url = 'https://www.uniprot.org/uploadlists/'

    params = {
        'from': source,
        'to': target,
        'format': 'tab',
        'query': query
    }

    data = urllib.parse.urlencode(params)
    data = data.encode('utf-8')
    req = urllib.request.Request(url, data)
    with urllib.request.urlopen(req) as f:
        response = f.read()

    df = pd.read_csv(
            io.StringIO(response.decode('utf-8')),
            sep='\t',
            index_col=0
        ).squeeze()

    df = df.loc[~df.index.duplicated(keep='first')]

    return df


if __name__ == '__main__':
    _, path = sys.argv
    prot = anndata.read(path)
    query = ' '.join(prot.var_names)
    response = translate_via_uniprot(query)

    shared = response.index.intersection(prot.var_names)
    prot.var.loc[shared, 'matched_ensg'] = response.loc[shared]
    prot.write(path)
