#!/bin/bash
set -e
source settings.sh

require_program "curl"
require_program "pigz"


# ====== Download and catalog assemblies from isolate sequencing project
echo "[*] Downloading assembly catalog."
curl -G "https://www.ebi.ac.uk/ena/portal/api/filereport" \
  -d 'result=assembly' \
  -d 'accession=PRJEB22252' \
  -d 'fields=accession,sample_accession,scientific_name,sample_title,assembly_name' \
  -d 'format=tsv' \
  -d 'limit=0' \
  -o ${ENA_ISOLATE_ASSEMBLY_CATALOG}


python ${BASE_DIR}/helpers/list_all_participants.py ${ENA_ISOLATE_ASSEMBLY_CATALOG} ${INFANT_ID_LIST}
