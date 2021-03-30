#!/bin/bash
set -e

# =====================================
# Change this to where project is located. Should be able to call `python scripts/run_inference.py`.
PROJECT_DIR="/mnt/f/microbiome_tracking"
# =====================================

# ======================================
# filesystem paths (relative to PROJECT_DIR) --> no need to modify.
BASE_DIR="${PROJECT_DIR}/examples/2strains"

CHRONOSTRAIN_INI="${BASE_DIR}/chronostrain.ini"
CHRONOSTRAIN_LOG_INI="${BASE_DIR}/logging.ini"
CHRONOSTRAIN_LOG_FILEPATH="${BASE_DIR}/logs/2strains.log"

READS_DIR="${BASE_DIR}/simulated_reads"
TRUE_ABUNDANCE_PATH="${BASE_DIR}/strain_abundances.csv"
OUTPUT_DIR="${BASE_DIR}/output"
PLOT_FORMAT="pdf"
INFERENCE_METHOD="em"
# =====================================

export BASE_DIR
export CHRONOSTRAIN_INI
export CHRONOSTRAIN_LOG_INI
export CHRONOSTRAIN_LOG_FILEPATH
mkdir -p $OUTPUT_DIR

# Sample reads.
python $PROJECT_DIR/scripts/simulate_reads.py \
--out_dir "${READS_DIR}" \
--abundance_path "${TRUE_ABUNDANCE_PATH}" \
--seed 123 \
--num_reads 500 \
--read_length 150

# Run inference algorithm.
# Time consistency on
python $PROJECT_DIR/scripts/run_inference.py \
--reads_dir $READS_DIR \
--true_abundance_path $TRUE_ABUNDANCE_PATH \
--method $INFERENCE_METHOD \
--read_length 150 \
--seed 123 \
-lr 0.001 \
--iters 3000 \
--out_dir $OUTPUT_DIR \
--plot_format $PLOT_FORMAT
