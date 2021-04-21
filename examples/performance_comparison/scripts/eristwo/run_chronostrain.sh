set -e
source settings.sh

# ===================================
# This script assumes the following variables:
# 1) Everything from settings.sh (already sourced)
# 2) CHRONOSTRAIN_LOG_FILEPATH
# 3) CHRONOSTRAIN_OUTPUT_DIR
# 4) SEED
# 5) READS_DIR
# ===================================

echo "Running inference."
python $PROJECT_DIR/scripts/run_inference.py \
--reads_dir ${READS_DIR}/filtered \
--true_abundance_path $TRUE_ABUNDANCE_PATH \
--method $CHRONOSTRAIN_METHOD \
--read_length $READ_LEN \
--seed $SEED \
-lr $CHRONOSTRAIN_LR \
--iters $CHRONOSTRAIN_NUM_ITERS \
--out_dir $CHRONOSTRAIN_OUTPUT_DIR \
--abundances_file $CHRONOSTRAIN_OUTPUT_FILENAME