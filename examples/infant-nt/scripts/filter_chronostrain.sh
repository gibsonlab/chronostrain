#!/bin/bash
set -e
source settings.sh

participant=$1
require_variable 'participant' $participant
require_file ${CHRONOSTRAIN_DB_JSON}

# =========== Run chronostrain. ==================
echo "Running filter on participant ${participant}."

run_dir=${DATA_DIR}/${participant}/chronostrain
export CHRONOSTRAIN_LOG_FILEPATH="${run_dir}/inference.log"
export CHRONOSTRAIN_CACHE_DIR="${run_dir}/.cache"
cd ${BASE_DIR}


env CUDA_VISIBLE_DEVICES='' chronostrain filter \
  -r ${run_dir}/reads.csv \
  -o ${run_dir}/filtered \
  --aligner "bwa-mem2"
touch ${run_dir}/filtered/FILTER_DONE.txt