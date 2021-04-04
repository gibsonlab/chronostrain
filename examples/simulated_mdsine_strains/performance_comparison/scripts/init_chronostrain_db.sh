#!/bin/bash

PROJECT_DIR="/PHShome/yk847/chronostrain"
CHRONOSTRAIN_DATA_DIR="/data/cctm/chronostrain"

# ======================================
# filesystem paths (relative to PROJECT_DIR) --> no need to modify.
BASE_DIR="${PROJECT_DIR}/examples/simulated_mdsine_strains/performance_comparison"

CHRONOSTRAIN_INI="${BASE_DIR}/files/chronostrain.ini"
CHRONOSTRAIN_LOG_INI="${BASE_DIR}/files/logging.ini"
CHRONOSTRAIN_LOG_FILEPATH="${CHRONOSTRAIN_DATA_DIR}/logs/readgen/db_init.log"
# =====================================

export BASE_DIR
export CHRONOSTRAIN_INI
export CHRONOSTRAIN_LOG_INI
export CHRONOSTRAIN_LOG_FILEPATH

# =========================================================
# Database initialization. (pre-download fasta and markers.)
python $BASE_DIR/scripts/initialize_chronostrain_database.py
# =========================================================