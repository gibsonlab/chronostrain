#!/bin/bash
set -e

source settings.sh
export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/init_pathogenic_genes.log"

echo "[*] Initializing database."
python ${BASE_DIR}/helpers/init_chronostrain_db.py \
--uniprot_tsv ${BASE_DIR}/files/pathogenic_markers.tsv \
-o ${CHRONOSTRAIN_DATA_DIR}/database_ecoli_pathogenic_all.json \
-r /mnt/d/ref_genomes

echo "[*] Pruning database by hamming similarity."
MULTIFASTA_FILE="all_strain_markers.fasta"
python ${BASE_DIR}/helpers/prune_chronostrain_db.py \
--input_json ${CHRONOSTRAIN_DATA_DIR}/database_ecoli_pathogenic_all.json \
--output_json ${CHRONOSTRAIN_DATA_DIR}/database_ecoli_pathogenic_pruned.json \
--alignments_path ${REFSEQ_ALIGN_PATH}
