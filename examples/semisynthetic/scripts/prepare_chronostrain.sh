#!/bin/bash
set -e
source settings.sh

echo "[*] Symlinking database structure..."
mkdir -p ${CHRONOSTRAIN_DB_DIR}
ln -s ${UMB_DB_DIR}/assemblies ${CHRONOSTRAIN_DB_DIR}/assemblies

echo "[*] Extracting ecoli-only database."
cp ${UMB_DB_DIR}/database_pruned_resolved.json ${CHRONOSTRAIN_DB_JSON}
python ${BASE_DIR}/helpers/extract_ecoli.py \
-i ${CHRONOSTRAIN_DB_JSON} \
-o ${ECOLI_ONLY_JSON}
