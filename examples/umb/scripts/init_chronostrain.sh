#!/bin/bash
set -e

source settings.sh
export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/init.log"

echo "[*] Initializing database."
python ${BASE_DIR}/helpers/init_chronostrain_db.py \
--metaphlan_pkl_path ${METAPHLAN_PKL_PATH} \
-o ${CHRONOSTRAIN_ECOLI_DB_JSON} \
-r /mnt/d/ref_genomes \
--use_local

echo "[*] Pruning database by hamming similarity."
MULTIFASTA_FILE="all_strain_markers.fasta"
python ${BASE_DIR}/helpers/prune_chronostrain_db.py \
--input_json ${CHRONOSTRAIN_ECOLI_DB_JSON} \
--output_json ${CHRONOSTRAIN_ECOLI_DB_JSON_PRUNED} \
--alignments_path ${REFSEQ_ALIGN_PATH}
