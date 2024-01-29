from itertools import product 

seeds = [117, 711, 420, 523, 187, 982, 766, 233, 523, 832]
subsets = ['intersection'] + [f'intersection_{seed}' for seed in seeds]
tumors = ['melanoma', 'ovarian']
datasets = ['rna', 'prot']
genesets = [
    'curated-melanoma-cell_type',
    'msigdb-c8.cell_type',
    'msigdb-hallmark',
    'all'
]
for subset, tumor, geneset in product(subsets, tumors, genesets):
    # filter tumor specific combinations
    if 'curated' in geneset and tumor != 'melanoma':
        continue

    # Unimodal for both RNA and Prot
    for dataset in datasets:
        cmd = f'python run_unimodal.py --tumor {tumor} --dataset {dataset} --geneset {geneset} --subset {subset}'

        if geneset == 'all':
            # can only run FA
            print(cmd, f'--method fa')
        else:
            print(cmd, f'--method plier')
            print(cmd, f'--method pathfa')

    # Multimodal
    cmd = f'python run_multimodal.py --tumor {tumor} --geneset {geneset} --subset {subset}'

    if geneset == 'all':
        print(cmd, f'--method mofa')
    else:
        print(cmd, f'--method pathfa')
