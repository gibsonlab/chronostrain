#!/bin/bash
set -e

source settings.sh
export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/filter.log"
SEED=31415

# =========== Read filtering. ===============

for umb_id in UMB05 UMB08 UMB11 UMB12 UMB15 UMB18 UMB20 UMB23 UMB24
do
	export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/filter_${umb_id}.log"

	echo "Filtering reads for ${umb_id}"
	python $PROJECT_DIR/scripts/filter_timeseries.py \
	--reads_input "${READS_DIR}/${umb_id}_inputs.csv" \
	-o "${READS_DIR}/${umb_id}_filtered" \
	--frac_identity_threshold 0.9 \
	--error_threhsold 0.05 \
	--num_threads 4
done
