#!/bin/bash
set -e
source settings.sh


mkdir -p ${DATA_DIR}/summary

export CHRONOSTRAIN_CACHE_DIR="${DATA_DIR}/cache"
export CHRONOSTRAIN_LOG_FILEPATH="${DATA_DIR}/summary/logs/evaluate.log"

python ${BASE_DIR}/scripts/quality_shift/eval_performance.py \
-b ${DATA_DIR} \
-g ${GROUND_TRUTH} \
-o ${DATA_DIR}/summary/output.csv