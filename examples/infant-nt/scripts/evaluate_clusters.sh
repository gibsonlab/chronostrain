#!/bin/bash
set -e
source settings.sh

require_program fastmlst
require_file ${CHRONOSTRAIN_DB_JSON}

# Run FastMLST to obtain ST calls for refseq assemblies.
mlst_result=${CHRONOSTRAIN_DB_DIR}/${CHRONOSTRAIN_DB_NAME}.mlst.tsv
echo "Targeting ${CHRONOSTRAIN_DB_DIR}/assemblies/*/*.fasta"
echo "Output of FastMLST will be written to: ${mlst_result}"
fastmlst -s '\t' ${CHRONOSTRAIN_DB_DIR}/assemblies/*/*.fasta > ${mlst_result}

# Evaluate chronostrain's marker-based database clusters in terms of fastMLST-annotated ST calls.
echo "Parsing output."
python ${BASE_DIR}/helpers/evaluate_clusters.py --json ${CHRONOSTRAIN_DB_JSON} --mlst ${mlst_result}
