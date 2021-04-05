#!/bin/bash
set -e

source settings.sh
export CHRONOSTRAIN_LOG_FILEPATH="${CHRONOSTRAIN_DATA_DIR}/logs/reads_${N_READS}/single-run/chronostrain.log"
SEED=31415

# =========== Run chronostrain. ==================
python $PROJECT_DIR/scripts/run_inference.py \
--reads_dir $READS_DIR \
--true_abundance_path $TRUE_ABUNDANCE_PATH \
--method $CHRONOSTRAIN_METHOD \
--read_length $READ_LEN \
--seed $SEED \
-lr $CHRONOSTRAIN_LR \
--iters $CHRONOSTRAIN_NUM_ITERS \
--out_dir $CHRONOSTRAIN_OUTPUT_DIR \
--abundances_file $CHRONOSTRAIN_OUTPUT_FILENAME
# ================================================
