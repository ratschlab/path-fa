# Code for "Probabilistic Pathway-based Multimodal Factor Analysis"

The main code for the PathFA method, which is implemented in `pathfa/path_fa.py` in lines
185 and 420 for the unimodal and multimodal variants. See the docstrings for more details.

## Setup and requirements
- Python >= 3.7 with installed requirements.txt and rpy linked
- msigdb v7.2 pathways/genesets when re-running data preparation
- TuPro datasets for reproducing the corresponding experiments

## Synthetic Experiments

The synthetic experiments use MSigDB Hallmark pathways and associations prepared in `data_synth`
to simulate example data.
To produce all commands that have to be run, execute `python runscripts/synthetic_joblist_generate.py > runscripts/synthetic_joblist.sh`.
Before executing each command, `mkdir synthetic_results` has to be executed as well.
Each line in the `synthetic_joblist.sh` corresponds to a command that has to be run.
For details on the synthetic experiment, see `run_synthetic.py`.
In `notebooks/`, use the ipynb notebook to produce figures.

# TuPro Experiments

The commands are generated with `python runscripts/joblist_tupro_generate.py > runscripts/joblist_tupro.sh` and each
line corresponds to a command that has to be executed.
Before executing all commands, make sure to create the directories `tupro_results/ovarian` and `tupro_results/melanoma`.
Further, make sure to set the variable `TUPRO_PATH` in `pathfa/utils.py` to the directory where to load TuPro data from (once it is released).
Afterwards, results are aggregated with `python scripts/extract_tupro_results.py`.
In `notebooks/`, use the ipynb notebook to produce figures.