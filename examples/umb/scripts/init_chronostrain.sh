#!/bin/bash
set -e

source settings.sh
export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/init.log"


echo "[*] Creating RefSeq index."
python ${BASE_DIR}/helpers/index_refseqs.py -r "/mnt/d/ref_genomes"


echo "[*] Creating Blast database."
REFSEQ_FASTA_FILE = ${BLAST_DB_DIR}/refseqs.fasta
> ${REFSEQ_FASTA_FILE}  # Clear file
for fasta_file in ${CHRONOSTRAIN_DB_DIR}/assemblies/*/*.fasta; do
	echo "Concatenating ${fasta_file}..."
	cat ${fasta_file} >> ${REFSEQ_FASTA_FILE}
done

cd ${BLAST_DB_DIR}
makeblastdb \
-in ${REFSEQ_FASTA_FILE} \
-out ${BLAST_DB_NAME} \
-dbtype nucl \
-title "Escherichia RefSeq Chromosomes" \
-parse_seqids

rm ${REFSEQ_FASTA_FILE}
cd -

echo "[*] Initializing database."
python ${BASE_DIR}/helpers/init_chronostrain_db.py \
-o ${CHRONOSTRAIN_ECOLI_DB_JSON} \
-dbdir ${BLAST_DB_DIR} \
-dbname ${BLAST_DB_NAME} \
--min_pct_idty 50 \
--max_target_seqs 100000 \
--metaphlan_pkl_path ${METAPHLAN_PKL_PATH} \
--reference_accession "U00096.3"


echo "[*] Pruning database by hamming similarity."
MULTIFASTA_FILE="all_strain_markers.fasta"
python ${BASE_DIR}/helpers/prune_chronostrain_db.py \
--input_json ${CHRONOSTRAIN_ECOLI_DB_JSON} \
--output_json ${CHRONOSTRAIN_ECOLI_DB_JSON_PRUNED} \
--alignments_path ${REFSEQ_ALIGN_PATH}
