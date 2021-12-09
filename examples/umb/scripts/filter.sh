#!/bin/bash
set -e

source settings.sh
export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/filter.log"
SEED=31415

# =========== Read filtering. ===============

for umb_id in UMB05 UMB08 UMB11 UMB12 UMB15 UMB18 UMB20 UMB23 UMB24
do
	index_filename = "${READS_DIR}/${umb_id}_${INPUT_INDEX_FILENAME}"

	echo "Filtering reads for ${umb_id}"
	cd $PROJECT_DIR/scripts
	python filter.py \
	-r "${READS_DIR}" \
	--input_file "${index_filename}" \
	-o "${READS_DIR}/${umb_id}_filtered" \
	--pct_identity_threshold 0.85 \
	--min_seed_length 10 \
	--num_threads 4
done
