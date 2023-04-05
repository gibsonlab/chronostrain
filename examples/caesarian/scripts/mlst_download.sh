#!/bin/bash
set -e
source settings.sh


cd ${BASE_DIR}
export CHRONOSTRAIN_LOG_FILEPATH="${OUTPUT_DIR}/logs/mlst_download.log"
python helpers/mlst_download.py \
  -o ${CHRONOSTRAIN_DB_DIR}/mlst.pkl \
  -t files/target_genera.txt
