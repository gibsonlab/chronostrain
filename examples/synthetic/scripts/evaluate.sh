#!/bin/bash
set -e
source settings.sh


mkdir -p ${DATA_DIR}/summary
export CHRONOSTRAIN_CACHE_DIR="${DATA_DIR}/cache"
python ${BASE_DIR}/helpers/eval_performance.py \
-b ${DATA_DIR} \
-g ${GROUND_TRUTH} \
-o ${DATA_DIR}/summary