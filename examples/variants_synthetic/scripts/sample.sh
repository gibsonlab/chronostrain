set -e
source settings.sh


variants_json="${BASE_DIR}/files/variants.json"
echo "Creating variants from ${variants_json}."

python ${BASE_DIR}/helpers/create_variants.py \
-i ${variants_json} \
-o ${CHRONOSTRAIN_DATA_DIR}/fasta


echo "Sampling reads."

python ${BASE_DIR}/helpers/sample_reads.py \
--fasta_dir ${CHRONOSTRAIN_DATA_DIR}/fasta \
--out_dir ${READS_DIR} \
--abundance_path ${TRUE_ABUNDANCE_PATH} \
--num_reads 10000000 \
--read_len 150 \
--profiles ${READ_PROFILE_PATH} ${READ_PROFILE_PATH} \
--num_cores $N_CORES \
--seed 31415 \
--clean_after_finish
