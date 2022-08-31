#!/bin/bash
set -e
source settings.sh

echo "[*] Symlinking database structure..."
mkdir -p ${CHRONOSTRAIN_DB_DIR}
ln -s ${UMB_DB_DIR}/assemblies ${CHRONOSTRAIN_DB_DIR}/assemblies

echo "[*] Extracting ecoli-only database."
python ${BASE_DIR}/helpers/init_db.py \
-i "${UMB_DB_DIR}/database_pruned_resolved.json" \
-o ${CHRONOSTRAIN_DB_JSON}
