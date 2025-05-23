# ChronoStrain

ChronoStrain is a bioinformatics tool for estimating abundance ratios of bacterial strains (e.g. ratio of E. coli strains or strain genome clusters).
As input, it takes a sequence of time-series, whole-genome metagenomic sequencing FASTQ files as input, and a reference database of known genomic variants of a particular species.

# Table of Contents
1. [Colab Demo](#colab-demo)
2. [Dependencies](#dependencies)
3. [Installation](#installation)
4. [Core Interface - Quickstart](#quickstart)
5. [Configuration](#config) 
6. [Reproducing paper analyses](#paper)
7. [Citing our work](#citing)

# 1. Colab Demo <a name="colab-demo"></a>

Navigate to the following link for a short demonstration (installation + usage)
<a href="https://colab.research.google.com/github/gibsonlab/chronostrain/blob/master/examples/colab_demo/demo.ipynb"><img alt="" src="https://img.shields.io/badge/Google%20Colab-Open%20tutorial-blue?style=flat&logo=googlecolab"/></a>


# 2. Dependencies <a name="dependencies"></a>

### A. Software
We require a working python installation (python >= 3.8, tested on python 3.10) or a working conda environment. We recommend mamba+miniforge: https://github.com/conda-forge/miniforge.

### B. Hardware
- For running the analysis, we recommend a CUDA-enabled NVIDIA GPU.
- Please allocate sufficient disk space for constructing databases. Even though the final database's disk usage is typically less than 500 MB, during construction & initialization, one may need more disk space. An Enterobacteriaeae-level complete assembly catalog used in our paper occupied ~70 GB of disk space after GZIP compression.
- Depending on read sequencing depth and database size, the analysis pipeline (`chronostrain advi`) may require more disk space to store intermediate bioinformatics results. For us, this meant up to 16G per time-series of disk space for E. faecalis (BBS analysis), and 1G per time-series of disk space for E. coli (UMB analysis).

# 3. Installation <a name="installation"></a>

Please pull this code from the git repository, via `git clone https://github.com/gibsonlab/chronostrain.git`

We provide a setup script for `pip` and also two different conda recipes. 
(Note: all cells below assume that the user is inside the git repo directory.)

### A. Basic conda recipe (Recommended, ~7G disk space, ~3 minutes)

Necessary dependencies only, installs cuda-toolkit from NVIDIA for (technically optional, but highly useful) GPU usage.
```bash
conda env create -f conda_basic.yml -n chronostrain
conda activate chronostrain
```

GPU usage is enabled by default. To disable it, set `JAX_PLATFORM_NAME=cpu` in the environment variables.

### B. Full conda recipe (~10G disk space, ~4 minutes)

Necessary dependencies + optional bioinformatics dependencies for running the examples/paper analyses.
(Note: this recipe also contains version numbers for packages that the paper analysis was run with.)

Includes additional dependencies such as: `sra-tools`, `kneaddata`, `trimmomatic` etc found in scripts in `example` 
subdirectories.

```bash
conda env create -f conda_full.yml -n chronostrain_paper
conda activate chronostrain_paper
```

### C. Pip

```bash
pip install .
```

Unlike the conda recipes, one needs to install the remaining dependencies separately. (`cuda-toolkit`, `bowtie2`, `bwa` and/or `bwa-mem2`, `blast`, `samtools`)


## 3.1 Other requirements

To enable database construction (`chronostrain make-db`), we require 
the tool <a href="https://github.com/dnbaker/dashing2">dashing2</a>.
Here is a link to the authors' repository for pre-built binaries: https://github.com/dnbaker/dashing2-binaries

After downloading and/or building an executable from source, the `dashing2` executable must be
discoverable; add it to your PATH environment variable.
One way to do this is to add the following to your `.bashrc`:

(*note: if/when dashing2 is added to a conda repository, we will add it to the conda recipe.*)
```bash
# assuming dashing2 binary is located in /home/username/dashing2_dir directory
export PATH=${PATH}:/home/username/dashing2_dir
```


# 4. Core Interface: Quickstart (Unix) <a name="quickstart"></a>

Installing chronostrain (using one of the above recipes) creates a command-line entry point `chronostrain`.

An example pipeline looks like the following:
```bash
# Step 0: create database (this only needs to be done once)
chronostrain make-db -m my_marker_seeds.tsv -r my_reference_genomes.tsv -b my_blast_db_name -bd tmp_blast_dir -o DB_DIR/my_database.json
chronostrain cluster-db -i DB_DIR/my_database.json -o DB_DIR/my_clusters.txt -t 0.998

# Step 1: filter reads
chronostrain filter -r timeseries_metagenomic_reads.tsv -o FILTERED_DIR -s DB_DIR/my_clusters.txt

# Step 2: perform inference
chronostrain advi -r FILTERED_DIR/filtered_timeseries_metagenomic_reads.tsv -o INFERENCE_DIR -s DB_DIR/my_clusters.txt
```

For precise I/O specification and a description of all arguments, please invoke the `--help` option.
Note that all commands below requires a valid configuration file; refer to [Configuration](#config).

1. **Database creation** -- **We HIGHLY recommend reading the notebook recipe `examples/database/complete_recipes/ecoli_mlst_simple.ipynb`.**   
    
    This command outputs a JSON file and populates a data directory specified by the configuration:
    ```bash
    chronostrain make-db -m <marker_seeds> -r <reference_catalog> -o <output_json> -b <blast_db_name> -db <blast_db_dir> ...
    ```
    The most important arguments are:
    - `-m, --marker-seeds FILE`: a TSV file of marker seeds,  with at least two columns: `[gene name], [path_to_fasta]` 
    - `-r, '--references FILE`: a TSV file that catalogs the reference genome collection. It must contain at least the following columns:
    `Accession`, `Genus`, `Species`, `Strain`, `ChromosomeLen`, `SeqPath`, `GFF`.
    The GFF column is optional, and is used to extract gene name annotations for metadata. 
    ChromosomeLen is only used as metadata; it is used in the "overall relative abundance" estimator.
    To download genomes, we recommend the `ncbi datasets` API.
    We provide example scripts that use this API, located in `examples/database/download_ncbi2.sh`

    Note that `blast_db_dir` points to a directory, and `blast_db_name` refers to a new blast database name. It will be created automatically.

    Then, one configures chronostrain to use the database
    ```text
    ...
    [Database.args]
    ENTRIES_FILE=<path_to_json>
    ...
    ```
    Refer to [Configuration](#config) for more details.
    An example json is included in `examples/example_configs/entero_ecoli.json`.
   
2. **Time-series read filtering**

    To run the pre-processing step for filtering a time-series collection of reads by aligning them to our database,
    run this command.
    ```bash
    chronostrain filter -r <read_tsv> -o <out_dir> [-s <cluster_txt>] ...
    ```
    The most important arguments here are:
    - `-r, --reads FILE`: The collection of reads from a time-series experiment is to be specified using TSV/CSV file (tab- or comma-separated), formatted in the following way
    (exclude headers). See below for the file format.
    - `-o, --out-dir DIRECTORY`: The directory to store the filtered reads into. This tool will also output a CSV file that catalogs the filtered reads.
    - `-s, --strain-subset FILE`: This is optional. 
    If specified, performs filtering only on the genome IDs listed in the file
    Typically, such a file will specify cluster representatives, generated by `chronostrain cluster-db`.
   
    **Sequencing read input file format (in CSV/TSV):**
    ```csv
    <timepoint>,<sample_name>,<experiment_read_depth>,<path_to_fastq>,<read_type>,<quality_fmt>
    ```
    - timepoint: A floating-point number specifying the timepoint annotation of the sample.
    - sample_name: A sample-specific description/name. Samples with the same sample_name will be grouped together as paired-end reads (user must specify `paired_1` and `paired_2` in `read_type`)
    - experiment_read_depth: The total number of reads sequenced (the "read depth") for this fastq file.
    - path_to_fastq: Path to the read FASTQ file (include a separate row for forward/reverse reads if using paired-end). Accepts relative paths (e.g. not starting with forward slash "/"). 
    GZIP is supported as well, if the filename ends with `.gz`.
    - read_type: one of `single`, `paired_1` or `paired_2`.
    - quality_fmt: Currently implemented options are: `fastq`, `fastq-sanger`, `fastq-illumina`, `fastq-solexa`. 
      Generally speaking, we can add on any format implemented in the `BioPython.QualityIO` module.
      
    This command outputs, into a directory of your choice, a collection of filtered FASTQ files, a summary of all alignments
    and a CSV file that catalogues the result (to be passed as input to `chronostrain advi`).
      
3. **Time-series inference**
    
    To run the inference using time-series reads that have been filered (via `chronostrain filter`), run this command.
    ```bash
    chronostrain advi -r <filtered_reads_tsv> -o <out_dir> [-s <cluster_txt>] ...
    ```
    The pre-requisite for this command is that one has run `chronostrain filter` to produce a TSV/CSV file that
    catalogues the filtered fastq files. Several options are available to tweak the optimization; we highly recommend
    that one enable the `--plot-elbo` flag to diagnose whether the stochastic optimization is converging properly.

4. **Abundance Profile Extraction**

    To use the estimated posteriors and extract a time-series abundance profile, run this command.
    ```bash
    chronostrain interpret -a <inference_out_dir> -r <filtered_reads_csv> -o <target_dir> -s <cluster_txt> [-p <pi_bar>] [-rs <species_name>] ...
    ```
    **The value of `-a, -r, -s` arguments must match what was passed into `chronostrain advi`.**
    The pre-requisite for this command is that one has run `chronostrain advi` which has estimated a posterior model,
    which has been saved to `<inference_out_dir>`. Note that this command samples from the conditional posterior, where
    only those database entries exceeding posterior inclusion probability $q(Z_s) > \bar{\pi}$ exceeding some 
    threshold $\bar{\pi}$ are included into the model.

    The output of this command is a `(T x N x S)` array (`<target_dir>/abundance_profile.npy`), where T is the number of timepoints, 
    S is the number of database clusters, and N is the target number of samples (N=5000 by default).
    For an example of how to use this command and its output, please refer to the colab notebook demo.
    


# 5. Configuration <a name="config"></a>

A configuration file for ChronoStrain is required, because it specifies parameters for our model, how many 
cores to use, where to store/load the database from, etc.

Configurations are specified by a file in the INI format; see `examples/example_configs/chronostrain.ini.example` for an example.

## First method: command-line
All subcommands can be preceded with the `-c / --config` argument, which specifies how the software is configured.

Usage:
```bash
chronostrain [-c CONFIG_PATH] <SUBCOMMAND>
```

Example:
```bash
chronostrain -c examples/example_configs/chronostrain.ini.example filter -r subject_1_timeseries.csv -o subj_1_filtered
```

## Second method: env variables

By default, ChronoStrain looks for the variable `CHRONOSTRAIN_INI`, if the `-c` option is not specified.
The following is equivalent to using the -c option from the above example:
```bash
export CHRONOSTRAIN_INI=examples/exmaple_configs/chronostrain.ini.example
chronostrain filter -r subject_1_timeseries.csv -o subj_1_filtered
```
in the scenario that both of the `-c` and `CHRONOSTRAIN_INI` are specified, the program will always 
prioritize the command-line argument.

## Optional: logging configuration

For debugging and/or requiring chronostrain to log more helpful and verbose status, specify the
`CHRONOSTRAIN_LOG_INI` variable.
```bash
export CHRONOSTRAIN_LOG_INI=./log_config.ini.example
```


# 6. Reproducing paper analyses <a name="paper"></a>

Please refer to the scripts/documentation found in the subdirectories in the companion repository: https://github.com/gibsonlab/chronostrain_paper 

- Semi-synthetic: `examples/semisynthetic`
- CAMI strain-madness challenge: separate repo -- `https://github.com/gibsonlab/chronostrain_CAMI`
- UMB Analysis: `examples/umb`
- BBS infant analysis: `examples/infant-nt`

# 7. Citing our work <a name="citing"></a>

To cite this software, please use the following publication reference:
Kim, Y., Worby, C.J., Acharya, S. et al. Longitudinal profiling of low-abundance strains in microbiomes with ChronoStrain. Nat Microbiol 10, 1184–1197 (2025). https://doi.org/10.1038/s41564-025-01983-z
