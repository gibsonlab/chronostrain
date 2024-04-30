#!/bin/bash
set -e
source settings.sh
source chronostrain/settings.sh

participant=$1
require_variable 'participant' $participant
require_file ${CHRONOSTRAIN_DB_JSON}

# =========== Run chronostrain. ==================
echo "[*] Running inference on participant ${participant}."

run_dir=${DATA_DIR}/${participant}/chronostrain_99_99pct
export CHRONOSTRAIN_LOG_FILEPATH="${run_dir}/precompute.log"
export CHRONOSTRAIN_CACHE_DIR="${DATA_DIR}/.cache"
cd ${BASE_DIR}


env JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu \
  chronostrain precompute \
  -r ${run_dir}/filtered/filtered_reads.csv \
  -s ${DATA_DIR}/database/chronostrain_files/efaecalis.clusters_99_99pct.txt
