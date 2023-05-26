#!/bin/bash
set -e
source settings.sh

# ----- This was the old database scheme using MLST-derived markers.
# ----- Refer to the assembly-based database creation instead. (initialize_db.sh)

cd ${BASE_DIR}
export CHRONOSTRAIN_LOG_FILEPATH="${OUTPUT_DIR}/logs/mlst_download.log"
export CHRONOSTRAIN_DB_NAME="mlst"

python helpers/mlst_download.py \
  -n mlst \
  -o mlst \
  -t files/target_genera.txt
