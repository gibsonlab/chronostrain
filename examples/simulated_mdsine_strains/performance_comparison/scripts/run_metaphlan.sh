#!/bin/bash
set -e

if [ -z ${PROJECT_DIR} ]; then
	echo "Variable 'PROJECT_DIR' is not set. Exiting."
	exit 1
else
	echo "PROJECT_DIR=${PROJECT_DIR}"
fi

# =====================================
# Command line args
NUM_READS=$1
TRIAL=$2

# ======================================
# filesystem paths (relative to PROJECT_DIR) --> no need to modify.
BASE_DIR="${PROJECT_DIR}/examples/simulated_mdsine_strains/performance_comparison"
CHRONOSTRAIN_DATA_DIR="/data/cctm/chronostrain"

RUNS_DIR="${CHRONOSTRAIN_DATA_DIR}/runs"
READ_LEN=150
TRIAL_DIR="${RUNS_DIR}/trials/reads_${NUM_READS}_trial_${TRIAL}"

READS_DIR="${TRIAL_DIR}/simulated_reads"
OUTPUT_DIR="${TRIAL_DIR}/output/metaphlan"
# =====================================

# =========== Run metaphlan. ==================
# TODO run metaphlan here.
# ================================================
