#!/bin/bash

PROJECT_DIR="/PHShome/yk847/chronostrain"

# ======================================
# filesystem paths (relative to PROJECT_DIR) --> no need to modify.
BASE_DIR="${PROJECT_DIR}/examples/simulated_mdsine_strains/performance_comparison"

DB_DIR="/data/cctm/chronostrain/metaphlan_db"

# =========================================================
echo "Initializing Metaphlan database."
echo "\tNote: Make sure chronostrain db is initialized first."
echo "\tTODO: remove this dependency by reading directly from chronostrain.ini."

mkdir -p $DB_DIR

