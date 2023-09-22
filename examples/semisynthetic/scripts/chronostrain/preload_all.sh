#!/bin/bash
set -e
source settings.sh
source chronostrain/settings.sh

cd ${BASE_DIR}/scripts

mkdir -p ${DATA_DIR}/databases/chronostrain
ln -s ${CHRONOSTRAIN_DB_DIR_SRC}/assemblies ${DATA_DIR}/databases/chronostrain/assemblies

env \
  JAX_PLATFORM_NAME=cpu \
  CHRONOSTRAIN_DB_JSON="${CHRONOSTRAIN_DB_JSON_SRC}" \
  CHRONOSTRAIN_DB_DIR="${DATA_DIR}/databases/chronostrain" \
  CHRONOSTRAIN_LOG_FILEPATH="${DATA_DIR}/databases/chronostrain/preload.log" \
  CHRONOSTRAIN_CACHE_DIR="./" \
  python chronostrain/preload_chronostrain_db.py
