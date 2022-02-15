#!/bin/bash
set -e

source settings.sh
export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/init.log"

echo "[*] Extracting markers from metaphlan database file."
python ${BASE_DIR}/helpers/extract_from_metaphlan.py \
-m ${METAPHLAN_PKL_PATH} \
-r ${NCBI_REFSEQ_DIR}

echo "[*] Initializing database."
python ${BASE_DIR}/helpers/init_chronostrain_db.py \
-o ${CHRONOSTRAIN_DB_JSON} \
-r ${NCBI_REFSEQ_DIR}

echo "[*] Pruning database by hamming similarity."
MULTIFASTA_FILE="all_strain_markers.fasta"
python ${BASE_DIR}/helpers/prune_chronostrain_db.py \
--input_json ${CHRONOSTRAIN_DB_JSON} \
--output_json ${CHRONOSTRAIN_DB_JSON_PRUNED} \
--alignments_path ${REFSEQ_ALIGN_PATH}
