#!/bin/bash
set -e

source settings.sh
export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/init.log"


python ${BASE_DIR}/helpers/init_chronostrain_db.py \
-m ${METAPHLAN_PKL_PATH} \
-o ${CHRONOSTRAIN_ECOLI_DB_JSON} \
-sdb /mnt/d/strainge/strainge_db

python ${BASE_DIR}/helpers/prune_chronostrain_db.py \
--input_json ${CHRONOSTRAIN_ECOLI_DB_JSON} \
--output_json ${CHRONOSTRAIN_ECOLI_DB_JSON_PRUNED} \
--alignments_path ${REFSEQ_ALIGN_PATH}
