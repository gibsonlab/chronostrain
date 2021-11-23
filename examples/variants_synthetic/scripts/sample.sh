set -e
source settings.sh


# =================================================
echo "Initializing chronostrain files."
export CHRONOSTRAIN_INI="${BASE_DIR}/files/chronostrain_init.ini"
python $BASE_DIR/helpers/initialize_chronostrain.py


echo "Creating variants from ${variants_json}."
variants_json="${BASE_DIR}/files/variants.json"
python ${BASE_DIR}/helpers/create_variants.py \
-v ${variants_json} \
-o ${CHRONOSTRAIN_DB_DIR}


# ==================================================
echo "Sampling reads."
export CHRONOSTRAIN_INI="${BASE_DIR}/files/chronostrain.ini"
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
