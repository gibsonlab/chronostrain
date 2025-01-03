# About

An analysis pipeline of the dataset provided by 2019 Shao et al.
- Dataset publication title: Stunted microbiota and opportunistic pathogen colonization in caesarean-section birth
  - DOI: https://doi.org/10.1038/s41586-019-1560-1
  - Link: https://www.nature.com/articles/s41586-019-1560-1

# Additional Requirements 

These are already included in the "full" conda recipe `conda_full.yml`.

- FastMLST (https://github.com/EnzoAndree/FastMLST)
- pigz
- KneadData
- (Requirements from `database` example)

# Important note about KneadData & Trimmomatic interaction on linux conda-forge setup

If one runs into the KneadData error that the jarfile for trimmomatic is invalid or corrupt, one needs to
edit `trimmomatic_jar=“trimmomatic*` to `trimmomatic_jar=trimmomatic.jar` in python script `<CONDA_ENV>/kneaddata/lib/<PYTHON>/site-packages/kneaddata/config.py` to explicitly point
to the jar file instead of the executable python wrapper.

# Pipeline walkthrough

All scripts are meant to be run from within the `scripts` subdirectory (so first do `cd scripts`)

Files are downloaded to the directory pointed to by the `DATA_DIR` environment variable, specified in `settings.sh`.

## 1. Download the dataset.

```bash
bash download_assembly_catalog.sh  # Download isolate assembly ENA metadata
bash download_metagenomic_catalog.sh  # Download metagenomic dataset ENA metadata
bash download_all.sh  # Download all participants' metagenomic reads for which an isolate exists, and run them through KneadData
```

Note that `download_all.sh` invokes `download_dataset.sh`, which handles downloading the reads from
a particular participant. It can be manually invoked to download a particular slice of the data (e.g. infant A01653):
```bash
# Assumes ENA metadata have been downloaded
bash download_dataset.sh A01653
bash ../helpers/process_dataset.sh A01653  # invoke kneaddata/trimmomatic pipeline.
```

## 2. Set up the database.
Use the provided `efaecalis` recipe in the `database` example. 
Ensure that it is configured to use the database directories that match this example.
Then, run the notebook "database_efaecalis_elmc.ipynb" in the `infant-nt/notebooks` subdirectory and execute all cells.


## 3. Perform analysis

There are two scripts to use; either will suit the purpose just fine.
To run the analysis using an already-downloaded dataset (e.g. assuming step 1 has finished), use the batch script
```bash
bash run_chronostrain_all.sh
```

Or, to perform analysis on a single participant (e.g. infant A01653)
```bash
bash run_chronostrain_all.sh A01653
```

To run the analysis concurrently while downloading the dataset from step 1, use the following script which 
periodically re-tries the entire cohort every 30 minutes as more participants are available locally on disk.
Note that this script (as well as `run_chronostrain_all.sh` leaves behind "breadcrumbs" such as 
`inference.DONE` to indicate which datasets have already been analyzed.)
```bash
bash run_all_watch.sh
```

All three scripts call the individual ingredients (e.g. `chronotrain filter`, `chronostrain advi`).

*Note: The analysis was originally done on an RTX 3090 which has 24GB of memory. 
If using a different GPU, one can tweak the memory usage requirements directly through `settings.sh`.
To reduce memory footprint, reduce `CHRONOSTRAIN_NUM_SAMPLES` and/or reduce `CHRONOSTRAIN_READ_BATCH_SZ`.*


## 4. Download assemblies (for partial validation)

```bash
bash download_assembly_catalog.sh  # only if you haven't done this already
bash download_assemblies.sh  # Runs a for loop to download all assemblies. Automatically invokes fastMLST.
```
The second script `download_assemblies.sh` automatically invokes fastMLST for ST annotation.
Results are stored separately in each participant's subdirectory.
