#!/bin/bash
set -e

source settings.sh
export CHRONOSTRAIN_LOG_FILEPATH="${CHRONOSTRAIN_DATA_DIR}/logs/umb/filter.log"
SEED=31415

# =========== Read filtering. ===============
echo "Filtering reads."
python ${PROJECT_DIR}/scripts/filter.py \
-r "${READS_DIR}" \
-o "${READS_DIR}/filtered"
