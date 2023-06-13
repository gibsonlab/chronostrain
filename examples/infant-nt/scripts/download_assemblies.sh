#!/bin/bash
set -e
source settings.sh

require_program "curl"
require_program "pigz"


# ====== Download and catalog assemblies from isolate sequencing project
ENA_ISOLATE_ASSEMBLY_CATALOG=${DATA_DIR}/isolate_assembly_ena.tsv
if ! [ -f ${ENA_ISOLATE_ASSEMBLY_CATALOG} ]
then
  echo "[*] Downloading assembly catalog."
  curl -G "https://www.ebi.ac.uk/ena/portal/api/filereport" \
    -d 'result=assembly' \
    -d 'accession=PRJEB22252' \
    -d 'fields=accession,sample_accession,scientific_name,sample_title,assembly_name' \
    -d 'format=tsv' \
    -d 'limit=0' \
    -o ${ENA_ISOLATE_ASSEMBLY_CATALOG}
fi


# ====== Initialize database objects.
python ${BASE_DIR}/helpers/download_assembly.py \
  -a ${ENA_ISOLATE_ASSEMBLY_CATALOG} \
  -o ${DATA_DIR}


# ====== run fastMLST.
for f in ${DATA_DIR}/*/isolate_assemblies; do
  cd $f
  fastmlst -s '\t' *.fasta > fastmlst.tsv
  cd -
done
