#!/bin/bash
set -e

source settings.sh
export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/filter.log"
SEED=31415

# =========== Read filtering. ===============
while IFS=, read -r patient
do
	echo "Filtering reads for ${patient}"
	python $PROJECT_DIR/scripts/filter_timeseries.py \
	--reads_input "${READS_DIR}/${patient}/${INPUT_INDEX_FILENAME}" \
	-o "${READS_DIR}/${patient}/filtered" \
	--pct_identity_threshold 0.85 \
	--num_threads 4
done < ${BASE_DIR}/files/patients.txt
