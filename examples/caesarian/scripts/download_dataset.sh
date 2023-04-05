#!/bin/bash
set -e
source settings.sh


curl \
  "https://www.ebi.ac.uk/ena/portal/api/filereport" \
  --get \
  -d 'accession=PRJEB32631' \
  -d 'result=read_run' \
  -d 'fields=sample_accession,scientific_name,fastq_ftp,sample_title,read_count' \
  -d 'format=tsv' \
  -d 'limit=0' \
  -o ${DATA_DIR}/project.tsv


cd ${BASE_DIR}
python helpers/dataset_download.py \
  -p ${DATA_DIR}/project.tsv \
  -o ${DATA_DIR}