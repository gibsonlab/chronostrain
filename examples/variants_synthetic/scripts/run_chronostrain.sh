#!/bin/bash
set -e

source settings.sh
export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/chronostrain.log"
SEED=31415

# =========== Run chronostrain. ==================
echo "Running inference."

cd $PROJECT_DIR/scripts
python inference_with_variants_simple.py \
--reads_dir "${READS_DIR}/filtered" \
--out_dir $CHRONOSTRAIN_OUTPUT_DIR \
--num_strands 4 \
--quality_format "fastq" \
--input_file "filtered_${INPUT_INDEX_FILENAME}" \
--seed $SEED \
--iters $CHRONOSTRAIN_NUM_ITERS \
--num_samples $CHRONOSTRAIN_NUM_SAMPLES \
--learning_rate $CHRONOSTRAIN_LR \
--plot_format "pdf"
# ================================================
