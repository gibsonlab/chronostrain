# ChronoStrain

## Package dependencies:
- torch>=1.7.0
- pandas>=1.1.3
- matplotlib
- seaborn
- joblib
- tqdm
- biopython

In addition, it requires either bwa or bowtie2.

## Installation/Configuration

As of 2/29/2021, a simple `pip install` suffices:

```bash
cd chronostrain
pip install .
```

Alternatively, one can include the cloned repository's directory to `PYTHONPATH` 
(at the cost of having to install the dependencies manually).

### Core Package: Quickstart (Unix)

By default, ChronoStrain automatically looks for two configuration files, `chronostrain.ini` and `log_config.ini` 
in the working directory.
This repository provides examples. For the most basic setup, copy them to their 
corresponding locations:

```bash
cp chronostrain.ini.example chronostrain.ini
cp log_config.ini.example log_config.ini
```

Alternatively, one can set the environment variables `CHRONOSTRAIN_INI` and/or `CHRONOSTRAIN_LOG_INI` to specify an arbitrary location.

```bash
export CHRONOSTRAIN_INI=/dir1/my_config.ini
export CHRONOSTRAIN_LOG_INI=/dir2/my_log_config.ini
```

More customization options are explained inside the `*.example` configurations. 

### Database configuration file

The database is specified using a particular JSON-formatted syntax, consisting of a single list.
Each entry of the list is a **Strain** object, which must contain three fields: `name`, `accession`, and `markers`.

1. `name`: A string containing a user-readable, familiar description of the strain. No restrictions; this is purely metadata.
2. `accession`: An accession number usable by NCBI.
3. `markers`: A list of JSON objects, each entry specifying a separate marker for that strain (a "Strain" is simply a collection of marker genes, representing its "signature".).
There are currently two types of markers supported: "tag" and "primer".
    - A "tag" marker is a simple annotated region of the genome, which must be present in the genbank page of the specified accession.
    The marker's `locus_id` is a locus tag found in NCBI's genbank annotation, and `name` is a user-defined identifier.
    - A "primer" marker is denoted using a forward and a reverse (flanking) sequence. The convention used here is that both are specified in the forward (5' to 3') direction.
    ChronoStrain will perform a matching search for both; if more than one is found, it takes the shortest one. (TODO copy number?)

--------------------------------
```json
[
    {
        "name": "bacteroides fragilis", 
        "accession": "CR626927.1",
        "markers": [
                    {"type":"tag", "locus_id":"BF9343_0009", "name":"glycosyltransferase"}, 
                    {"type":"tag", "locus_id":"BF9343_3272", "name":"hydrolase"}
                ]
    },
    {
        "markers": [
                    {"type":"tag", "locus_id":"ECOLIN_01070", "name":"16S"},
                    {"type": "primer", "forward": "GTGCCAGCMGCCGCGGTAA", "reverse": "GGACTACHVGGGTWTCTAAT", "name": "16S_V4"}
                ],
        "name": "escherichia coli nissle 1917",
        "accession": "CP007799.1"
    }
]
```
--------------------------------
