# ChronoStrain


# Table of Contents
1. [Colab Demo](#colab-demo)
2. [Installation](#installation)
3. [Core Interface - Quickstart](#quickstart)
4. [Configuration](#config) 
5. [Manually defining a database](#manual-db)
   1. [Strain Definition](#strain-def)
   2. [Marker Sequence Definition](#marker-def)
   3. [Source Sequence Definition](#seq-def)
6. [Reproducing paper analyses](#paper)

# 1. Colab Demo <a name="colab-demo"></a>

(WORK IN PROGRESS, UNDER CONSTRUCTION) <a href="https://colab.research.google.com/github/gibsonlab/chronostrain/blob/master/examples/colab_demo/demo.ipynb"><img alt="" src="https://img.shields.io/badge/Google%20Colab-Open%20tutorial-blue?style=flat&logo=googlecolab"/></a>

# 2. Installation <a name="installation"></a>

There are three ways to install chronostrain.

### A. Basic conda recipe (Recommended)

Necessary dependencies only, installs cuda-toolkit from NVIDIA for (optional, but highly useful) GPU usage.
```bash
conda env create -f conda_basic.yml -n chronostrain
conda activate chronostrain
```

GPU usage is enabled by default. To disable it, set `JAX_PLATFORM_NAME=cpu` in the environment variables.

### B. Full conda recipe

Necessary dependencies + optional bioinformatics dependencies for running the examples/paper analyses.

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

one may need to pick and choose the proper pytorch version beforehand (e.g. with/without cuda).


## 2.1 Other requirements

To enable database construction (`chronostrain make-db`), we require 
the tool <a href="https://github.com/dnbaker/dashing2">dashing2</a>.
After downloading and/or building an executable from source, the `dashing2` executable must be
discoverable; add it to your PATH environment variable.
One way to do this is to add the following to your `.bashrc`:

(*note: if/when dashing2 is added to a conda repository, we will add it to the conda recipe.*)
```bash
# assuming dashing2 binary is located in /home/username/dashing2_dir directory
export PATH=${PATH}:/home/username/dashing2_dir
```


# 3. Core Interface: Quickstart (Unix) <a name="quickstart"></a>

Installing chronostrain creates a command-line entry point `chronostrain`.

An example pipeline looks like the following:
```bash
# Step 0: create database (this only needs to be done once)
chronostrain make-db -m my_marker_seeds.tsv -r my_reference_genomes.tsv -o DB_DIR/my_database.json
chronostrain cluster-db -i DB_DIR/my_database.json -o DB_DIR/my_clusters.txt -t 0.998

# Step 1: filter reads
chronostrain filter -r timeseries_metagenomic_reads.tsv -o FILTERED_DIR -s DB_DIR/my_clusters.txt

# Step 2: perform inference
chronostrain advi -r FILTERED_DIR/filtered_timeseries_metagenomic_reads.tsv -o INFERENCE_DIR -s DB_DIR/my_clusters.txt
```

For precise I/O specification and a description of all arguments, please invoke the `--help` option.
Note that all commands below requires a valid configuration file; refer to [Configuration](#config).

1. **Database creation** -- **We recommend following the notebook recipe `examples/database/complete_recipes/<example>.ipynb`.**   
    
    If one really wants to generate a new database manually, this command outputs a JSON file and populates a data directory 
    specified by the configuration:
    ```bash
    chronostrain make-db -m <marker_seeds> -r <reference_catalog> -o <output_json> ...
    ```
    The most important arguments are:
    - `-m, --marker-seeds FILE`: a TSV file of marker seeds,  with at least two columns: `[gene name], [path_to_fasta]` 
    - `-r, '--references FILE`: a TSV file that catalogs the reference genome collection. It must contain at least the following columns:
    `Accession`, `Genus`, `Species`, `Strain`, `ChromosomeLen`, `SeqPath`, `GFF`.
    The GFF column is optional, and is used to extract gene name annotations for metadata. 
    ChromosomeLen is only used as metadata; it is used in the "overall relative abundance" estimator.
    To download genomes, we recommend the `ncbi datasets` API.
    We provide example scripts that use this API, located in `examples/database/download_ncbi2.sh`

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


# 4. Configuration <a name="config"></a>

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

# 5. Defining a database manually <a name="manual-db"></a>

In general, one may refer to our manuscript for how we define strains using BLAST and a 
cleaning procedure, resulting in a .json strain definition file.

For those who wish to implement scripts to make a database differently (e.g. without using
`chronostrain make-db`), one only needs to adhere to the following convention.

The JSON file ought to be formatted as a single list of objects:
```text
[
    { ...strain1 },
    { ...strain2 }
]
```

## Strain Definition <a name="strain-def"></a>

A strain is simply a collection of marker sequences, plus some metadata 
(species/genus/strain name/chromosome length (if known)).

Example of a strain definition:
```text
{
    "id": "STRAIN_ID",
    "genus": "Escherichia",
    "species": "coli",
    "name": "K12_MG1655",
    "genome_length": 4631469,
    "seqs": [
        {"id": "SEQ_ID_1", "seq_path": "path_to_fasta"}
    ],
    "markers": [
        {
            "id": "UNIQUE_GENE_ID",
            "name": "COMMON_GENE_NAME" ,
            "type": "subseq",
            "source": "SEQ_ID_1",
            "source_i": FASTA_RECORD_IDX,
            "start": 840273,
            "end": 840855,
            "strand": "+"
        }
    ]
}
```
All of the listed fields are required (to enforce clean/interpretable database records), but not all are used.

## Marker Sequence Definition <a name="marker-def"></a>

A strain-specific marker sequence is defined using the following attributes.

- **id**: a string that uniquely identifies this sequence (no two markers, even between different strain 
  entries, may share an ID)
- **name**: a common name for the sequence. A good choice is the name of a gene, if it represents one.
  (*NOTE: This is just metadata that helps one organize how the marker sequence was chosen.*)
- **type**: A string specifying how you want to extract the subsequence from the chromosome. Currently, only "subseq" is supported.
- **source**: A FASTA file from which this marker is a subsequence of.
- **source_i**: The index of the FASTA record. Example: to index into the first record, pass `"source_i": 0`.
- **start**:  an integer encoding the starting (inclusive) position on the chromosome.
- **end**: an integer, encoding the last (inclusive) position on the chromosome.
- **strand**: either `"+"` or `"-"` (quotes included). 
  If minus, the database will first extract the start -> end position substring, *and then*
  compute the reverse complement of the subsequence.


## Source Sequence Definition <a name="seq-def"></a>

Each strain entry contains a `seqs` field, which specifies the source sequence that each marker should be pulled out of.
Each corresponds to a FASTA file `<accession>.fasta`, containing at least one FASTA sequence record representing the 
assembled genome (either contigs or a completed assembly. Scaffolds with gaps are not accepted.).
If these files are missing (in the `DATA_DB_DIR` directory, as specified in the configuration), then ChronoStrain 
can be configured to automatically download these files from NCBI on-the-fly.
(If this behavior is something that you would like, please use a valid NCBI-searchable accession for the 
`accession` field, and enable NCBI API usage in the configuration by setting ENABLED=True under "[Entrez]".)

In the case of a strain with a *complete* chromosomal assembly, one only needs to provide a single-object list:
```text
...
"seqs": [{"accession": "SEQ_ID_1", "seq_path": "path_to_fasta"}],
...
```

If a complete assembly is not available and one only has contigs (it can be split into multiple fasta files), one can specify multiple sources:
```text
...
"seqs": [
    {"accession": "CONTIG_ACC_1", "seq_path": "path_to_fasta1"},
    {"accession": "CONTIG_ACC_2", "seq_path": "path_to_fasta2"},
    ...
],
...
```

# 6. Reproducing paper analyses <a name="paper"></a>

Please refer to the scripts/documentation found in each respective subdirectory.

- Semi-synthetic: `examples/semisynthetic`
- UMB Analysis: `examples/umb`
- ELMC infant analysis: `examples/infant-nt` 
