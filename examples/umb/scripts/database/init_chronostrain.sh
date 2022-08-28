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
cd ${BASE_DIR}/helpers
python init_chronostrain_db.py \
-o ${CHRONOSTRAIN_DB_JSON_ALL} \
-dbdir ${BLAST_DB_DIR} \
-dbname ${BLAST_DB_NAME} \
--min_pct_idty 75 \
--refseq_index ${NCBI_REFSEQ_DIR}/index.tsv \
--uniprot_csv ${BASE_DIR}/files/uniprot_markers.tsv \
--genes_fasta ${BASE_DIR}/files/clermont_genes.fasta \
--metaphlan_pkl ${METAPHLAN4_PKL_PATH} \
--metaphlan_pkl ${METAPHLAN3_PKL_PATH} \
--reference_accession "U00096.3"


echo "[*] Resolving overlaps."
python validate_database.py \
--refseq_index ${NCBI_REFSEQ_DIR}/index.tsv \
-i ${CHRONOSTRAIN_DB_JSON_ALL} \
-o ${CHRONOSTRAIN_DB_JSON_RESOLVED}


echo "[*] Pruning database by hamming similarity."
MULTIFASTA_FILE="all_strain_markers.fasta"
cd ${BASE_DIR}/helpers
python concatenated_multiple_alignments.py --raw_json ${CHRONOSTRAIN_DB_JSON_ALL} --align_path ${MULTI_ALIGN_PATH}
python prune_chronostrain_db.py \
--raw_json ${CHRONOSTRAIN_DB_JSON_ALL} \
--merged_json ${CHRONOSTRAIN_DB_JSON_RESOLVED} \
--output_json ${CHRONOSTRAIN_DB_JSON_PRUNED} \
--align_path ${MULTI_ALIGN_PATH}
