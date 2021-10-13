#!/bin/bash
set -e

source settings.sh
export CHRONOSTRAIN_LOG_FILEPATH="${CHRONOSTRAIN_DATA_DIR}/logs/umb/chronostrain.log"
SEED=31415

# =========== Run chronostrain. ==================
echo "Running inference."

cd $PROJECT_DIR/scripts
python inference_with_variant_search.py \
--reads_dir "${READS_DIR}/filtered" \
--read_length $READ_LEN \
--out_dir $CHRONOSTRAIN_OUTPUT_DIR \
--quality_format "fastq" \
--input_file "input_files.csv" \
--seed $SEED \
--iters $CHRONOSTRAIN_NUM_ITERS \
--num_samples $CHRONOSTRAIN_NUM_SAMPLES \
--learning_rate $CHRONOSTRAIN_LR \
--abundances_file $CHRONOSTRAIN_OUTPUT_FILENAME \
--plot_format "pdf"
# ================================================
