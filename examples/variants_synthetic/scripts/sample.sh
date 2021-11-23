set -e
source settings.sh

# ==================================================
echo "Sampling reads."
export CHRONOSTRAIN_INI="${BASE_DIR}/files/chronostrain.ini"
python ${BASE_DIR}/helpers/sample_reads.py \
--out_dir ${READS_DIR} \
--abundance_path ${TRUE_ABUNDANCE_PATH} \
--num_reads 10000000 \
--read_len 150 \
--profiles ${READ_PROFILE_PATH} ${READ_PROFILE_PATH} \
--num_cores $N_CORES \
--seed 31415 \
--clean_after_finish
