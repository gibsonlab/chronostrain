#!/bin/bash
set -e

source settings_singlegene.sh
export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/filter.log"
SEED=31415

# =========== Read filtering. ===============

for umb_id in UMB05 UMB08 UMB11 UMB12 UMB15 UMB18 UMB20 UMB23 UMB24
do
	echo "Filtering reads for ${umb_id}"
	python $PROJECT_DIR/scripts/filter_single.py \
	--reads_input "${READS_DIR}/${umb_id}_${INPUT_INDEX_FILENAME}" \
	-o "${READS_DIR}/${umb_id}_filtered_singlegene" \
	--pct_identity_threshold 0.85 \
	--num_threads 4
done