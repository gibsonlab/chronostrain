#!/bin/bash

PROJECT_DIR="/mnt/f/microbiome_tracking"
CHRONOSTRAIN_DATA_DIR="/home/younhun/data"

# ======================================
# filesystem paths (relative to PROJECT_DIR) --> no need to modify.
BASE_DIR="${PROJECT_DIR}/examples/simulated_mdsine_strains/performance_comparison"

CHRONOSTRAIN_INI="${BASE_DIR}/files/chronostrain_for_metaphlan.ini"
CHRONOSTRAIN_LOG_INI="${BASE_DIR}/files/logging.ini"
CHRONOSTRAIN_LOG_FILEPATH="${CHRONOSTRAIN_DATA_DIR}/logs/readgen/db_init.log"
DB_DIR="${CHRONOSTRAIN_DATA_DIR}/metaphlan_converted_db"
# =========================================================

echo "============== Initializing Metaphlan database. =============="
echo "	Note: Make sure chronostrain db is initialized first."
echo "	TODO: remove this dependency by reading directly from chronostrain.ini."

export BASE_DIR
export CHRONOSTRAIN_INI
export CHRONOSTRAIN_LOG_INI
export CHRONOSTRAIN_LOG_FILEPATH
mkdir -p $DB_DIR
cd ${DB_DIR}

python /mnt/f/microbiome_tracking/examples/simulated_mdsine_strains/performance_comparison/scripts/initialize_metaphlan_database.py \
--input_path /home/younhun/metaphlan_db/mpa_v30_CHOCOPhlAn_201901.pkl \
--out_dir $DB_DIR \
--basename mpa_chronostrain
