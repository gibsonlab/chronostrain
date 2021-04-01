# Example: Bacterial strains (large test)

Benchmarking chronostrain (with filtering) on sampled reads, using strains from the original MDSINE paper.

## Running this example

This script is meant to be run on the ERISONE compute cluster, which employs the LSF platform for job 
submission & resource allocation.

### Step 1: Master script

1) Open `scripts/run_master_lsf.sh` using a text editor, and edit the variable PROJECT_DIR at the top.
2) **Log into a compute node**.
3) Run this script using bash:

```
bash scripts/run_master_lsf.sh
```

This script does four things:
- Pre-download the necessary files for strain marker database.
- Sample reads for each *(n_reads, trial index)* pair.
- Generate two LSF files for each *(n_reads, trial_index)* pair, one for chronostrain and one for MetaPhlan.
- Submit each LSF using `bsub`.


### Step 2: Plotting

TODO
