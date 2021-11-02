#!/bin/bash
set -e

source settings.sh
export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/filter.log"
SEED=31415

# =========== Read filtering. ===============
echo "Filtering reads."
cd $PROJECT_DIR/scripts
python filter.py \
-r "${READS_DIR}" \
--input_file "${INPUT_INDEX_FILENAME}" \
-o "${READS_DIR}/filtered" \
--pct_identity_threshold 0.3 \
--min_seed_length 10 \
--num_threads 4