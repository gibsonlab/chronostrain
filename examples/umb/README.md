# Requirements: 

## Before you start 

This example requires the following to be run first:
1) Installation of MetaPhlAn (v3.0)
2) Installation of the `ncbi-genome-download` tool.
3) The locations of these installations edited into `settings.sh`

In addition, it requires the installation of these programs (most are available on bioconda):
- sra-tools=2.11
- blast
- trimmomatic
- gzip

and these python packages:
- bioservices (for UniProt access)

Optionally, for deriving the E.coli clades, we used ClermonTyping.
Its command-line inputs can be programmatically generated using `bash helpers/create_clermontyping_input.sh`

# Order of scripts to run

Run the bash scripts below in the given order.

1. `download_database.sh`: Download all listed Escherichia strains from NCBI, using ncbi-genome-download. 
2. `download_samples.sh`: Download all samples related to the UMB study (BioProject PRJNA400628) from NCBI.
3. `process_samples.sh`: Pre-process each sample using trimmomatic, to remove adapters.
4. `init_chronostrain.sh`: Initialize chronostrain DB from metaphlan-derived markers, using strains filtered out by StrainGE's tutorial step.
5. `filter.sh`: The alignment-based filtering step of chronostrain.
6. `run_chronostrain.sh`: The (non de-novo) inference, on the constructed database.

*(Note: The individual sequence accessions from BioProject PRJNA400628 were derived using the script `helpers/fetch_records.py`)*