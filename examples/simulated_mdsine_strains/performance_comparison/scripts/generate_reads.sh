#!/bin/bash
set -e

# =====================================
# Change this to where project is located. Should be able to call `python scripts/run_inference.py`.
PROJECT_DIR="/mnt/f/microbiome_tracking"
# =====================================
# Command line args
NUM_READS=$1
TRIAL=$2


# ======================================
# filesystem paths (relative to PROJECT_DIR) --> no need to modify.
BASE_DIR="${PROJECT_DIR}/examples/em_perf"  # TODO change this when moving this back to simulated_mdsine_strains.

CHRONOSTRAIN_INI="${BASE_DIR}/chronostrain.ini"
CHRONOSTRAIN_LOG_INI="${BASE_DIR}/logging.ini"
CHRONOSTRAIN_LOG_FILEPATH="${BASE_DIR}/logs/read_depth_${NUM_READS}/run.log"

TRUE_ABUNDANCE_PATH="${BASE_DIR}/default/true_abundances.csv"  # TODO change this.

# Where to store the summary CSV file of all trials.
OUTPUT_DIR="${BASE_DIR}/output"
PLOT_FORMAT="pdf"

READ_LEN=150
# =====================================

export BASE_DIR
export CHRONOSTRAIN_INI
export CHRONOSTRAIN_LOG_INI
export CHRONOSTRAIN_LOG_FILEPATH

RUN_BASEDIR="${OUTPUT_DIR}/read_depth_${NUM_READS}"
READS_DIR="${RUN_BASEDIR}/simulated_reads"
RUN_OUTPUT_DIR="${RUN_BASEDIR}/output"
mkdir -p $READS_DIR
mkdir -p $RUN_OUTPUT_DIR

OUTPUT_FILENAME="abundances.out"
SEED=$trial

# Run the trials.
echo "[Number of reads: ${n_reads}, trial #${trial}]"

TRIAL_DIR="${OUTPUT_DIR}/trials/reads_${n_reads}_trial_${trial}"
READS_DIR="${TRIAL_DIR}/simulated_reads"
mkdir -p $READS_DIR

SEED=$trial

# ============ Generate the reads. ===============
# TODO replace this with Zack's sampler.
# Generate the reads.
python $PROJECT_DIR/scripts/simulate_reads.py \
--seed $SEED \
--out_dir $READS_DIR \
--abundance_path $TRUE_ABUNDANCE_PATH \
--num_reads $NUM_READS \
--read_length $READ_LEN
# ================================================

