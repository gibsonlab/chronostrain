#!/bin/bash
set -e
source settings.sh


python ${BASE_DIR}/helpers/init_db.py \
-i "/mnt/d/chronostrain/umb_database/database_pruned.json" \
-o ${CHRONOSTRAIN_DB_JSON}