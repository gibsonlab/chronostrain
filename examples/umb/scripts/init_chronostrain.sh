#!/bin/bash
set -e

source settings.sh
export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/init.log"


python ${BASE_DIR}/helpers/create_db_from_uniref.py \
-u ${BASE_DIR}/files/gene_info_uniref.csv \
-s ${STRAINGE_STRAIN_LIST} \
-o ${CHRONOSTRAIN_ECOLI_DB_SPEC} \
-sdb /mnt/d/strainge/strainge_db
