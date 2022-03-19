#!/bin/bash
set -e

source settings.sh
export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/init_MLST_genes.log"

#echo "[*] Downloading BLAST database."
#cd ${BLAST_DB_DIR}
#update_blastdb.pl nt \
#--source aws \
#--blastdb_version 5 \
#--num_threads 4 \
#--verbose
#cd -

echo "[*] Initializing database."
cd ${BASE_DIR}/helpers
python init_chronostrain_db.py \
-o ${CHRONOSTRAIN_DATA_DIR}/database_ecoli_MLST_all.json \
-dbdir ${BLAST_DB_DIR} \
-dbname ${BLAST_DB_NAME} \
--min_pct_idty 75 \
--max_target_seqs 1000000 \
--uniprot_csv ${BASE_DIR}/files/mlst_markers.csv \
--reference_accession "U00096.3"

echo "[*] Pruning database by hamming similarity."
MULTIFASTA_FILE="all_MLST_markers.fasta"
python prune_chronostrain_db.py \
--input_json ${CHRONOSTRAIN_DATA_DIR}/database_ecoli_MLST_all.json \
--output_json ${CHRONOSTRAIN_DATA_DIR}/database_ecoli_MLST_pruned.json \
--alignments_path ${REFSEQ_ALIGN_PATH}
