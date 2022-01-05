#!/bin/bash
set -e

source settings.sh
export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/init.log"


python ${BASE_DIR}/helpers/init_chronostrain_db.py \
-m ${METAPHLAN_PKL_PATH} \
-s ${STRAINGE_STRAIN_LIST} \
-o ${CHRONOSTRAIN_ECOLI_DB_SPEC} \
-sdb /mnt/d/strainge/strainge_db
