#!/bin/bash
set -e
source settings.sh

participant=$1

if [ -z "$participant" ]
then
	echo "var \"participant\" is empty"
	exit 1
fi


METAGENOMIC_CATALOG="${DATA_DIR}/project.tsv"
if ! [ -f ${METAGENOMIC_CATALOG} ]
then
  echo "[*] Downloading metagenomic reads catalog."
  curl \
    "https://www.ebi.ac.uk/ena/portal/api/filereport" \
    --get \
    -d 'accession=PRJEB32631' \
    -d 'result=read_run' \
    -d 'fields=sample_accession,scientific_name,fastq_ftp,sample_title,read_count' \
    -d 'format=tsv' \
    -d 'limit=0' \
    -o ${METAGENOMIC_CATALOG}
fi


cd ${BASE_DIR}
python helpers/dataset_download.py \
  -m ${METAGENOMIC_CATALOG} \
  -p ${participant} \
  -o ${DATA_DIR}