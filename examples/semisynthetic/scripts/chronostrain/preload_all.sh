#!/bin/bash
set -e
source settings.sh

cd ${BASE_DIR}/scripts

for (( replicate = 1; replicate < ${N_GENOME_REPLICATES}+1; replicate++ )); do
  replicate_dir=$(get_replicate_dir "$replicate")
  env \
    CHRONOSTRAIN_DB_JSON="${replicate_dir}/databases/chronostrain/ecoli.json" \
    CHRONOSTRAIN_DB_DIR="${replicate_dir}/databases/chronostrain" \
    CHRONOSTRAIN_LOG_FILEPATH="${replicate_dir}/databases/chronostrain/preload.log" \
    CHRONOSTRAIN_CACHE_DIR="./" \
    python chronostrain/preload_chronostrain_db.py
done