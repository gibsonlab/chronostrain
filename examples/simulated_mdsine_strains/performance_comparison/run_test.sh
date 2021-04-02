#!/bin/bash
set -e

# =====================================
# Change this to where project is located. Should be able to call `python scripts/run_inference.py`.
PROJECT_DIR="/mnt/f/microbiome_tracking"
# =====================================
# The location of the ReadGenAndFiltering repo for sampling
READGEN_DIR=""
# A TSV mapping from designation to reference genome file path for the sampler. 
# Designations must match those in true_abundances
COMMUNITY_REFERENCES=""
# =====================================
# Command line args
NUM_READS=$1
N_TRIALS=$2

# ======================================
# filesystem paths (relative to PROJECT_DIR) --> no need to modify.
BASE_DIR="${PROJECT_DIR}/examples/simulated_mdsine_strains/performance_comparison"

CHRONOSTRAIN_INI="${BASE_DIR}/chronostrain.ini"
CHRONOSTRAIN_LOG_INI="${BASE_DIR}/logging.ini"
CHRONOSTRAIN_LOG_FILEPATH="${BASE_DIR}/logs_${NUM_READS}reads/run.log"

TRUE_ABUNDANCE_PATH="${BASE_DIR}/true_abundances.csv"

# Where to store the summary CSV file of all trials.
OUTPUT_DIR="${BASE_DIR}/output"
PLOT_FORMAT="pdf"

READ_LEN=150
METHOD="em"
# =====================================
export BASE_DIR
export CHRONOSTRAIN_INI
export CHRONOSTRAIN_LOG_INI
export CHRONOSTRAIN_LOG_FILEPATH

# Run the trials.
for trial in $(seq 1 $N_TRIALS)
do
  echo "[Number of reads: ${NUM_READS}, trial #${trial}]"

  TRIAL_DIR="${OUTPUT_DIR}/trials/reads_${NUM_READS}_trial_${trial}"
  READS_DIR="${TRIAL_DIR}/simulated_reads"
  TRIAL_OUTPUT_DIR="${TRIAL_DIR}/output"
  mkdir -p $READS_DIR
  mkdir -p $TRIAL_OUTPUT_DIR

  OUTPUT_FILENAME="abundances.out"
  SEED=$trial

  # ============ Generate the reads. ===============
  python $READGEN_DIR/ReadGen.py \
  $NUM_READS $READ_LEN $trial \
  $READGEN_DIR/profiles/HiSeqReference \
  $READGEN_DIR/profiles/HiSeqReference \
  $COMMUNITY_REFERENCES \
  $TRUE_ABUNDANCE_PATH \
  $READS_DIR \
  $SEED
  # ================================================

  # =========== Run chronostrain. ==================
  python $PROJECT_DIR/scripts/run_inference.py \
  --reads_dir $READS_DIR \
  --true_abundance_path $TRUE_ABUNDANCE_PATH \
  --method $METHOD \
  --read_length $READ_LEN \
  --seed $SEED \
  -lr 0.001 \
  --iters 3000 \
  --out_dir $TRIAL_OUTPUT_DIR \
  --abundances_file $OUTPUT_FILE \
  --plot_format $PLOT_FORMAT
  # ================================================
done
