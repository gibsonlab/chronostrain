# ChronoStrain

# Table of Contents
1. [Installation](#installation)
   1. [Quickstart](#quickstart)
2. [Configuration](#config)
3. [Manually defining a database](#manual-db)
    1. [Strain Definition](#strain-def)
    2. [Marker Sequence Definition](#marker-def)

# 1. Installation <a name="installation"></a>

As of 2/29/2021, a simple `pip install` suffices:

```bash
cd chronostrain
pip install .
```

We will release configuration files for a conda environment in a near-future update.

## Core Interface: Quickstart (Unix) <a name="quickstart"></a>

Installing chronostrain using `pip` creates a command-line entry point `chronostrain`.
For precise i/o specification and argument description, please invoke the `--help` option.


1. **Database creation** (if starting from scratch)
   
   This command outputs a JSON file and populates a data directory 
   specified by the configuration:
    ```bash
    chronostrain make-db <ARGS>
    ```
    This has a major prerequisite: one needs to have constructed a directory of reference sequences, and catalogued this repository via
    a TSV file, to be passed via the `--reference` argument.
    The TSV file must contain at least the following columns:
    `Accession`, `Genus`, `Species`, `Strain`, `ChromosomeLen`, `SeqPath`, `GFF`.
    An easy way to do this is using the `ncbi-genome-download` tool and using our script (link here).
   
2. **Time-series read filtering**

    To run the pre-processing step for filtering a time-series collection of reads by aligning them to our database,
    run this command.
    ```bash
    chronostrain filter <ARGS>
    ```
   A pre-requisite for this command is that one has a CSV file, formatted in the following way:
    ```csv
    <timepoint>,<experiment_read_depth>,<path_to_fastq>,<read_type>,<quality_fmt>
    ```
    - timepoint: A floating-point number specifying the timepoint annotation of the sample.
    - experiment_read_depth: The total number of reads sequenced (the "read depth") from the sample.
    - path_to_fastq: Path to the read FASTQ file (include a separate row for forward/reverse reads if using paired-end)
    - read_type: one of `single`, `paired_1` or `paired_2`.
    - quality_fmt: Currently implemented options are: `fastq`, `fastq-sanger`, `fastq-illumina`, `fastq-solexa`. 
      Generally speaking, we can add on any format implemented in the `BioPython.QualityIO` module.
   


# 2. Configuration <a name="config"></a>

A configuration file for ChronoStrain is required, because it specifies parameters for our model, how many 
cores to use, where to store/load the database from, etc.

Configurations are specified by a file in the INI format; see `chronostrain.ini.example` for an example.

## First method: command-line
All subcommands inherit the `-c / -config` argument, which specifies how the software is configured.

Usage:
```bash
chronostrain [-c CONFIG_PATH] <SUBCOMMAND>
```

Example:
```bash
chronostrain filter -c ./chronostrain.ini.example -r subject_1_timeseries.csv -o subj_1_filtered
```

## Second method: env variables

By default, ChronoStrain uses the variable `CHRONOSTRAIN_INI`.
The following is equivalent to using the -c option from the above example:
```bash
export CHRONOSTRAIN_INI=./chronostrain.ini.example
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

# 3. Defining a database manually <a name="manual-db"></a>

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
    "seqs": [{"accession": "SEQ_ID_1", "seq_type": "chromosome"}],
    "markers": [
        {
            "id": "UNIQUE_GENE_ID",
            "name": "COMMON_GENE_NAME (e.g. fimA)" ,
            "type": "subseq",
            "source": "SEQ_ID_1",
            "start": 4532932,
            "end": 4533480,
            "strand": "+",
            "canonical": true
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
- **type**: A string specifying how you want to extract the subsequence from the chromosome. 
  (see below for implemented choices)
- **source**: A FASTA file from which this marker is a subsequence of.
- **canonical**: (not currently used, please specify "true". It is a carry-over from some experimental 
  features that is not yet complete.)

The implemented *type* options are:

1. `subseq`: specify a marker sequence as a position-specific subsequence of a source contig/chromosome.
   
    Usage:
    ```text
    {
        "id": "my_id", "name": "my_name",
        "type": subseq,
        "start": START_POSITION,
        "end": END_POSITION,
        "strand": "PLUS_or_MINUS"
    }
    ```
    - `start`:  an integer encoding the starting (inclusive) position on the chromosome.
    - `end`: an integer, encoding the last (inclusive) position on the chromosome.
    - `strand`: either `"+"` or `"-"` (quotes included). 
      If minus, the database will first extract the start -> end position substring, *and then*
      compute the reverse complement of the subsequence.
    

2. `primer`: Perform an exact pair of matches on the source contig/chromosome, and extract the subsequence flanked 
   by the two matches.
   
    Usage:
    ```text
    {
        "id": "my_id", "name": "my_name",
        "type": subseq,
        "forward": "ACCGGTGCCT",
        "reverse": "CGATTTTCTT"
    }
    ```
3. `locus_tag`: Using GenBank annotation, extract the (unique) entry which matches the locus_tag. 
   (the `source` accession's FASTA file must be accompanied by a genbank annotation file `<accession>.gb`)
   
    Usage:
    ```text
    {
        "id": "my_id", "name": "my_name",
        "type": subseq,
        "locus_tag": "<GENBANK_LOCUS_TAG>"
    }
    ```

Note that using either of the `primer` or `locus_tag` types won't account for possible subsequences with 
high % identity elsewhere in any of the chromosomes (say, 75% nucleotide identity after alignment).
The user is responsible for checking beforehand if one cares about non-primer matching or non-annotated regions that fit 
this criteria.

(*NOTE: In a future update, we will include an explicit "fasta" option to allow the user to **directly**
specify the marker sequence.*)

## Source Sequence Definition

Each strain entry contains a `seqs` field, which specifies the source sequence that each marker should be pulled out of.

In the case of a strain with a *complete* chromosomal assembly, one only needs to provide a single-object list:
```text
...
"seqs": [{"accession": "SEQ_ID_1", "seq_type": "chromosome"}],
...
```
when using the `locus_tag` type and one does *not* have a genbank annotation file on hand, one needs to
define the `accession` field using an NCBI nucleotide accession (ChronoStrain will automatically download
the genbank file from NCBI using the accession string.)

If a complete assembly is not available and one only has scaffolds or contigs, one can specify multiple sources:
```text
...
"seqs": [
    {"accession": "SCAFFOLD_ACC_1", "seq_type": "scaffold"},
    {"accession": "SCAFFOLD_ACC_2", "seq_type": "scaffold"},
    {"accession": "CONTIG_ACC_1", "seq_type": "contig"},
    {"accession": "CONTIG_ACC_2", "seq_type": "contig"}
],
...
```
Note that the `primer` option's search will fail if no scaffold or contig contains
*both* forward and reverse primer matches.
