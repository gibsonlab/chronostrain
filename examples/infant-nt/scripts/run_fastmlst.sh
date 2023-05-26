#!/bin/bash
set -e
source settings.sh

# Run FastMLST on UMB database.

ST_OUTPUT=${CHRONOSTRAIN_DB_DIR}/fastmlst.tsv
echo "Targeting ${CHRONOSTRAIN_DB_DIR}/assemblies/*/*.fasta"
echo "Output of FastMLST will be written to: ${ST_OUTPUT}"

fastmlst -s '\t' ${CHRONOSTRAIN_DB_DIR}/assemblies/*/*.fasta > ${ST_OUTPUT}
