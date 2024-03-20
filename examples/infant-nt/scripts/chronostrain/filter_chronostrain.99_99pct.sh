#!/bin/bash
set -e
source settings.sh

participant=$1
require_variable 'participant' $participant
require_file ${CHRONOSTRAIN_DB_JSON}

# =========== Run chronostrain. ==================
echo "[*] Running filter on participant ${participant}."

run_dir=${DATA_DIR}/${participant}/chronostrain_99_99pct
export CHRONOSTRAIN_LOG_FILEPATH="${run_dir}/filter.log"
export CHRONOSTRAIN_CACHE_DIR="${DATA_DIR}/.cache"
cd ${BASE_DIR}


env JAX_PLATFORM_NAME=cpu JAX_PLATFORM=cpu \
  chronostrain filter \
  -r ${run_dir}/reads.csv \
  -o ${run_dir}/filtered \
  -s ${DATA_DIR}/database/chronostrain_files/efaecalis.clusters_99_99pct.txt \
  --aligner "bwa-mem2"
touch ${run_dir}/filtered/FILTER_DONE.txt
