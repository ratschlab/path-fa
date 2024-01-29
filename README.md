# TuPro Experiments

The commands are generated with `runscripts/joblist_generate.py` and `runscripts/joblist_cv_generate.py`.
The first is to generate the commands for all methods and all the samples and the second does 10 random subsampling of the samples so that we can bootstrap error bars for the deconvolution performance.
To compute the final performance scores in terms of deconvolution, we use `scripts/extract_cytof_correlation.py` to get the final mean and standard errors in the case of the `cv`.
In the case without cv, only the mean is valid.

# Synthetic Experiments

Similar setup but with `runscripts/synthetic_joblist_generate.py` that already samples multiple datasets and therefore no leaving out samples for cv is necessary.
The results are directly available and can be analyzed.

## Setup

- install requirements.txt with Python >= 3.7 (running version atm)
- used msigdb v7.2 pathways/genesets

## Reproduce synthetic
- mkdir synthetic_results
- python runscipts/joblist_synthetic_generate.py > runscripts/joblist_synthetic.sh
- mkdir logs
- sbatch synthetic_job.sh

## Reproduce TuPro
- mkdir tupro_results
- mkdir tupro_results/ovarian
- mkdir tupro_results/melanoma
- python runscripts/joblist_tupro_generate.py > runscripts/joblist_tupro.sh
- sbatch tupro_job.sh
- python scripts/extract_tupro_results.py

## Reproduce figures
- open notebooks/figures_synthetic_and_tupro.ipynb and execute the cells after running all experiments
