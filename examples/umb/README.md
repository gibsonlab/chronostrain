# Requirements: 

## Before you start 

This example requires the following to be run first:
1) Installation of MetaPhlAn (v3.0)
2) Installation and initialization of StrainGE database.
3) The locations of these installations edited into `settings.sh`

In addition, it requires the installation of these libraries:
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

1. `init_chronostrain.sh`: Initialize chronostrain DB from metaphlan-derived markers, using strains filtered out by StrainGE's tutorial step.
2. `download_samples.sh`: Download all samples related to the UMB study (BioProject PRJNA400628) from NCBI.
3. `process_samples.sh`: Pre-process each sample using trimmomatic, to remove adapters.
4. `filter.sh`: The alignment-based filtering step of chronostrain.
5. `run_chronostrain.sh`: The (non de-novo) inference, on the constructed database.

*(Note: The individual sequence accessions from BioProject PRJNA400628 were derived using the script `helpers/fetch_records.py`)*