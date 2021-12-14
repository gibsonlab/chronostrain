#!/bin/bash
set -e

source settings.sh
export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/chronostrain.log"
SEED=31415

# =========== Run chronostrain. ==================
echo "Running inference."

cd $PROJECT_DIR/scripts

for umb_id in UMB05 UMB08 UMB11 UMB12 UMB15 UMB18 UMB20 UMB23 UMB24
do
	index_filename="${umb_id}_${INPUT_INDEX_FILENAME}"

	python inference_with_variants_simple.py \
	--reads_dir "${READS_DIR}/${umb_id}_filtered" \
	--out_dir $CHRONOSTRAIN_OUTPUT_DIR/${umb_id} \
	--quality_format "fastq" \
	--input_file "index_filename" \
	--seed $SEED \
	--iters $CHRONOSTRAIN_NUM_ITERS \
	--num_samples $CHRONOSTRAIN_NUM_SAMPLES \
	--learning_rate $CHRONOSTRAIN_LR \
	--plot_format "pdf"
done
# ================================================
