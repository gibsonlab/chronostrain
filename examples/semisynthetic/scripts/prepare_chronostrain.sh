#!/bin/bash
set -e
source settings.sh

echo "[*] Symlinking database structure..."
ln -s ${UMB_DB_DIR}/assemblies ${CHRONOSTRAIN_DB_DIR}/assemblies

echo "[*] Extracting ecoli-only database."
python ${BASE_DIR}/helpers/init_db.py \
-i "${UMB_DB_DIR}/database_pruned.json" \
-o ${CHRONOSTRAIN_DB_JSON}
