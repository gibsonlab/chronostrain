#!/bin/bash

PROJECT_DIR="/PHShome/yk847/chronostrain"
CHRONOSTRAIN_DATA_DIR="/data/cctm/chronostrain"

# ======================================
# filesystem paths (relative to PROJECT_DIR) --> no need to modify.
BASE_DIR="${PROJECT_DIR}/examples/simulated_mdsine_strains/performance_comparison"

CHRONOSTRAIN_INI="${BASE_DIR}/files/chronostrain.ini"
CHRONOSTRAIN_LOG_INI="${BASE_DIR}/files/logging.ini"
CHRONOSTRAIN_LOG_FILEPATH="${CHRONOSTRAIN_DATA_DIR}/logs/readgen/db_init.log"
DB_DIR="${CHRONOSTRAIN_DATA_DIR}/metaphlan_db"
# =========================================================

echo "============== Initializing Metaphlan database. =============="
echo "\tNote: Make sure chronostrain db is initialized first."
echo "\tTODO: remove this dependency by reading directly from chronostrain.ini."

export BASE_DIR
export CHRONOSTRAIN_INI
export CHRONOSTRAIN_LOG_INI
export CHRONOSTRAIN_LOG_FILEPATH
mkdir -p $DB_DIR

python ${BASE_DIR}/scripts/initialize_metaphlan_database.py \
--input_path /data/cctm/chronostrain/default_metaphlan_db/mpa_v30_CHOCOPhlAn_201901.pkl \
--out_dir $DB_DIR \
--basename mpa_chronostrain
