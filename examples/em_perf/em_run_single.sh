#!/bin/bash
set -e

# =====================================
# Change this to where project is located. Should be able to call `python scripts/run_inference.py`.
PROJECT_DIR="/mnt/f/microbiome_tracking"
# =====================================

# ======================================
# filesystem paths (relative to PROJECT_DIR) --> no need to modify.
BASE_DIR="${PROJECT_DIR}/examples/em_perf"

CHRONOSTRAIN_INI="${BASE_DIR}/chronostrain.ini"
CHRONOSTRAIN_LOG_INI="${BASE_DIR}/logging.ini"
CHRONOSTRAIN_LOG_FILEPATH="${BASE_DIR}/logs/em_perf.log"

TRUE_ABUNDANCE_PATH="${BASE_DIR}/default/true_abundances.csv"

OUTPUT_DIR="${BASE_DIR}/output"
READS_DIR="${OUTPUT_DIR}/simulated_reads"
PLOT_FORMAT="pdf"

READ_LEN=150
NUM_READS=200
METHOD="em"
SEED=123
# =====================================
export BASE_DIR
export CHRONOSTRAIN_INI
export CHRONOSTRAIN_LOG_INI
export CHRONOSTRAIN_LOG_FILEPATH


mkdir -p $READS_DIR
OUTPUT_FILENAME="abundances.out"

# Generate the reads.
python $PROJECT_DIR/scripts/simulate_reads.py \
--seed $SEED \
--out_dir $READS_DIR \
--abundance_path $TRUE_ABUNDANCE_PATH \
--num_reads $NUM_READS \
--read_length $READ_LEN

# Run chronostrain.
python $PROJECT_DIR/scripts/run_inference.py \
--reads_dir $READS_DIR \
--true_abundance_path $TRUE_ABUNDANCE_PATH \
--method $METHOD \
--read_length $READ_LEN \
--seed $SEED \
-lr 0.001 \
--iters 3000 \
--out_dir $OUTPUT_DIR \
--abundances_file $OUTPUT_FILENAME \
--skip_filter
