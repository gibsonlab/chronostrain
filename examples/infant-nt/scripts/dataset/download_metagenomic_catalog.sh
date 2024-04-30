#!/bin/bash
set -e
source settings.sh


mkdir -p ${DATA_DIR}
echo "[*] Downloading metagenomic reads catalog."
curl \
  "https://www.ebi.ac.uk/ena/portal/api/filereport" \
  --get \
  -d 'accession=PRJEB32631' \
  -d 'result=read_run' \
  -d 'fields=sample_accession,scientific_name,fastq_ftp,sample_title,read_count' \
  -d 'format=tsv' \
  -d 'limit=0' \
  -o ${DATASET_METAGENOMIC_CATALOG}
