#!/bin/bash
set -e
source ../scripts/settings.sh
export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/create_clermontyping_input.log"


python create_clermontyping_input.py \
-i ${STRAINGE_DB_DIR} \
-c ~/ClermonTyping/clermonTyping.sh \
-o ${CHRONOSTRAIN_DB_DIR}/ClermonTyping/run_clermontyping.sh \
