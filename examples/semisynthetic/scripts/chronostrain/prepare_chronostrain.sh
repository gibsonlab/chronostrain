#!/bin/bash
set -e
source settings.sh

#echo "[*] Symlinking database structure..."
#mkdir -p ${CHRONOSTRAIN_DB_DIR}
#ln -s ${UMB_DB_DIR}/assemblies ${CHRONOSTRAIN_DB_DIR}/assemblies
#
#echo "[*] Extracting ecoli-only database."
#cp ${UMB_DB_DIR}/database_pruned_resolved.json ${CHRONOSTRAIN_DB_JSON}
#python ${BASE_DIR}/helpers/extract_ecoli.py \
#-i ${CHRONOSTRAIN_DB_JSON} \
#-o ${ECOLI_ONLY_JSON}


python chronostrain/prepare_chronostrain_db.py \
  -i ${CHRONOSTRAIN_DB_JSON_SRC} \
  -o ${CHRONOSTRAIN_DB_JSON} \
  -s NZ_CP092452.1 \
  -m NZ_CP092452.1.sim_mutant \
  -s NZ_CP024859.1 \
  -m NZ_CP024859.1.sim_mutant \
  -g ${DATA_DIR}/sim_genomes \
  -ds ${CHRONOSTRAIN_DB_DIR_SRC} \
  -dt ${CHRONOSTRAIN_DB_DIR}

