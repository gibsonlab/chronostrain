#!/bin/bash
set -e
source settings.sh


env \
  JAX_PLATFORM_NAME=cpu \
  CHRONOSTRAIN_DB_JSON="${CHRONOSTRAIN_DB_JSON}" \
  CHRONOSTRAIN_DB_DIR="${CHRONOSTRAIN_DB_DIR}" \
  CHRONOSTRAIN_CACHE_DIR="./" \
  python preload_chronostrain_db.py
