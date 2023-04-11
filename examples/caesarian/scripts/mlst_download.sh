#!/bin/bash
set -e
source settings.sh

# ----- This was the old database scheme using MLST-derived markers.
# ----- Refer to the assembly-based database creation instead. (initialize_db.sh)

cd ${BASE_DIR}
export CHRONOSTRAIN_LOG_FILEPATH="${OUTPUT_DIR}/logs/mlst_download.log"
python helpers/mlst_download.py \
  -o ${CHRONOSTRAIN_DB_DIR}/mlst.pkl \
  -t files/target_genera.txt
