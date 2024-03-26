#!/bin/bash
set -e
source settings.sh

require_program "curl"
require_program "pigz"
require_file $ENA_ISOLATE_ASSEMBLY_CATALOG


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
