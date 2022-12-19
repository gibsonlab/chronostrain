#!/bin/bash
set -e

source settings.sh
export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/init.log"


echo "[*] Creating RefSeq index."
python ${BASE_DIR}/helpers/index_refseqs.py -r ${NCBI_REFSEQ_DIR}


echo "[*] Creating Blast database."
REFSEQ_FASTA_FILE=${BLAST_DB_DIR}/refseqs.fasta
echo "Target fasta file: ${REFSEQ_FASTA_FILE}"

mkdir -p ${BLAST_DB_DIR}
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
-title "Enterobacteriaceae RefSeq Chromosomes" \
-parse_seqids

rm ${REFSEQ_FASTA_FILE}
cd -

echo "[*] Initializing database."
python ${BASE_DIR}/helpers/init_chronostrain_db.py \
-o ${CHRONOSTRAIN_DB_JSON_ALL} \
-dbdir ${BLAST_DB_DIR} \
-dbname ${BLAST_DB_NAME} \
--min_pct_idty 75 \
--refseq_index ${NCBI_REFSEQ_DIR}/index.tsv \
--uniprot_csv ${BASE_DIR}/files/uniprot_markers.tsv \
--genes_fasta ${BASE_DIR}/files/clermont_genes.fasta \
--metaphlan_pkl ${METAPHLAN3_PKL_PATH} \
--reference_accession "U00096.3"


echo "[*] Pruning database by hamming similarity."
python ${BASE_DIR}/helpers/concatenated_multiple_alignments.py --raw_json ${CHRONOSTRAIN_DB_JSON_ALL} --align_path ${MULTI_ALIGN_PATH}
python ${BASE_DIR}/helpers/prune_chronostrain_db.py \
--source_json ${CHRONOSTRAIN_DB_JSON_ALL} \
--output_json ${CHRONOSTRAIN_DB_JSON_PRUNED} \
--align_path ${MULTI_ALIGN_PATH} \
--identity_threshold 0.002


echo "[*] Resolving overlaps."
python ${BASE_DIR}/helpers/validate_database.py \
--refseq_index ${NCBI_REFSEQ_DIR}/index.tsv \
-i ${CHRONOSTRAIN_DB_JSON_PRUNED} \
-o ${CHRONOSTRAIN_DB_JSON_RESOLVED}
