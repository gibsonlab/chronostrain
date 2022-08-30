#!/bin/bash
set -e
source settings.sh

log_dir=${DATA_DIR}/summary
export CHRONOSTRAIN_LOG_FILEPATH="${log_dir}/evaluate.log"
export CHRONOSTRAIN_CACHE_DIR=.
python ${BASE_DIR}/helpers/evaluate.py \
-b ${DATA_DIR} \
-o ${DATA_DIR}/summary \
-g ${RELATIVE_GROUND_TRUTH}

#python ${BASE_DIR}/helpers/roc_curves.py \
#-b ${DATA_DIR} \
#-i ${REFSEQ_INDEX} \
#-o ${DATA_DIR}/summary \
#-g ${RELATIVE_GROUND_TRUTH}
