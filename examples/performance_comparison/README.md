# Example: Bacterial strains (large test)

Benchmarking chronostrain (with filtering) on sampled reads, using strains from the original MDSINE paper.

## Running this example

This script is meant to be run on the ERISONE compute cluster, which employs the LSF platform for job 
submission & resource allocation.

### Step 0: Prerequisites

This workflow expects a conda environment called `chronostrain`, with package `chronostrain` installed.
It also assumes that `bowtie2` and `metaphlan 3.0` is installed:

```
conda install -c bioconda bowtie2
conda install -c bioconda metaphlan
```

### Step 1: Master script

1) Open `scripts/run_master_lsf.sh` using a text editor, and edit the variable PROJECT_DIR at the top. 
(Absolute pathing is preferable, but relative pathing also works.)
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

TODO modify plot.sh to draw boxplot, not lineplot with error bars.


# Strains chosen:

Akkermansia muciniphila
t__GCA_002885715

Bacteroides fragilis
t__GCA_000599365

Escherichia coli
t__GCA_000401755

Bacteroides ovatus
t__GCA_000273215

Bacteroides vulgatus
t__GCA_000403235

Clostridium sporogenes
t__GCA_001597905

Enterococcus faecalis
t__GCA_002945655

Klebsiella oxytoca
t__GCA_900083575

Parabacteroides distasonis
t__GCA_000699785

Proteus mirabilis
t__GCA_001858185

Ruminococcus gnavus
t__GCA_002865465