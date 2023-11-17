#!/bin/bash
set -e
source settings.sh

participant=$1
require_variable 'participant' $participant
require_file ${CHRONOSTRAIN_DB_JSON}

# =========== Run chronostrain. ==================
echo "[*] Running evidence quantification on participant ${participant}."

run_dir=${DATA_DIR}/${participant}/chronostrain
export CHRONOSTRAIN_LOG_FILEPATH="${run_dir}/evidence.log"
export CHRONOSTRAIN_CACHE_DIR="${run_dir}/.cache"
cd ${BASE_DIR}

env chronostrain quantify \
  -r ${run_dir}/filtered/filtered_reads.csv \
  -o ${run_dir}/inference \
  -s ${CHRONOSTRAIN_CLUSTER_FILE} \
