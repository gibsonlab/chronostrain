#!/bin/bash
set -e
source settings.sh

check_program "curl"
check_program "pigz"


participant=$1

if [ -z "$participant" ]
then
	echo "var \"participant\" is empty"
	exit 1
fi

# ====== Download and catalog assemblies from isolate sequencing project
ENA_ISOLATE_ASSEMBLY_CATALOG=${DATA_DIR}/isolate_assembly_ena.tsv
if ! [ -f ${ENA_ISOLATE_ASSEMBLY_CATALOG} ]
then
  echo "[*] Downloading assembly catalog."
  curl -G "https://www.ebi.ac.uk/ena/portal/api/filereport" \
    -d 'result=assembly' \
    -d 'accession=PRJEB22252' \
    -d 'fields=accession,sample_accession,scientific_name' \
    -d 'format=tsv' \
    -d 'limit=0' \
    -o ${ENA_ISOLATE_ASSEMBLY_CATALOG}
fi


# ====== Download sample metadata.
ENA_ISOLATE_READS_CATALOG=${DATA_DIR}/isolate_reads_ena.tsv
if ! [ -f ${ENA_ISOLATE_READS_CATALOG} ]
then
  echo "[*] Downloading sample catalog."
  curl \
    "https://www.ebi.ac.uk/ena/portal/api/filereport" \
    -G \
    -d 'result=read_run' \
    -d 'accession=PRJEB22252' \
    -d 'fields=sample_accession,sample_title' \
    -d 'format=tsv' \
    -d 'limit=0' \
    -o ${ENA_ISOLATE_READS_CATALOG}
fi


# ====== Initialize database objects.
export CHRONOSTRAIN_DB_SPECIFICATION=""
export CHRONOSTRAIN_DB_NAME=""
python ${BASE_DIR}/helpers/initialize_db.py \
  -a ${ENA_ISOLATE_ASSEMBLY_CATALOG} \
  -r ${ENA_ISOLATE_READS_CATALOG} \
  -p ${participant} \
  -o ${DATA_DIR}
