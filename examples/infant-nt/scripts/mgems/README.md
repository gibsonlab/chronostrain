# READ ME BEFORE YOU RUN THIS

The scripts in this sub-directory runs the hierarchical pipeline for mGEMS using
`demix_check` (https://github.com/harry-thorpe/demix_check). 

## Set up the database

This must be done prior to running any of the scripts.
In your favorite directory [DB_DIR], you must create the following database files:
```text
directory [DB_DIR]:
    [DB_DIR]/ref_dir/
    [DB_DIR]/ref_file.tsv

directory [DB_DIR]/ref_dir:
    [DB_DIR]/ref_dir/Efaecalis/
    [DB_DIR]/ref_dir/species_ref/
    
directory [DB_DIR]/ref_dir/species_ref:
    [DB_DIR]/ref_dir/species_ref/ref_info.tsv

directory [DB_DIR]/ref_dir/Efaecalis:
    [DB_DIR]/ref_dir/Efaecalis/ref_info.tsv
```
and run `demix_check --mode_setup` to compile the pseudoalignment indices (this is key, to follow the auto-generated naming conventnions.)

These are all files that adhere to the `demix_check` format.
Refer to that repository's README file, section `Modes -- run` which describes the hierarchical run configuration.
For our paper, we set up the species-level bins using the concatenation of:
- Enterobacteriaceae catalog from `UMB` and `semisynthetic` examples -- `ecoli_db/ref_genomes/index.tsv`
- Enterococcaceae catalog from `infant_nt` (the parent example of this subdir) -- `infant_nt/database/ref_genomes/index.tsv`
- A download of A. baumanii, P. aeruginosa, S. aureus, S. pneumoniae using the `download_ncbi.py` script from the `database` example.

The Efaecalis-level bins were calculated using the E. faecalis genomes and running PopPUNK (`infant-nt/scripts/msweep/run_poppunk.sh`).
