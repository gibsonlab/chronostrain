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
TRIALS_INDEX_PATH="${OUTPUT_DIR}/trials_index.csv"
PLOT_FORMAT="pdf"

READ_LEN=150
N_TRIALS=10
METHOD="em"
# =====================================
export BASE_DIR
export CHRONOSTRAIN_INI
export CHRONOSTRAIN_LOG_INI
export CHRONOSTRAIN_LOG_FILEPATH

# Erase contents of trials.csv
mkdir -p $OUTPUT_DIR
> $TRIALS_INDEX_PATH

# Run the trials.
for n_reads in 10 30 50 70 90 100 200 300 400 500 600 700 800 900
do
  for trial in $(seq 1 $N_TRIALS)
  do
    echo "[Number of reads: ${n_reads}, trial #${trial}]"

    TRIAL_DIR="${OUTPUT_DIR}/trials/reads_${n_reads}_trial_${trial}"
    READS_DIR="${TRIAL_DIR}/simulated_reads"
    TRIAL_OUTPUT_DIR="${TRIAL_DIR}/output"
    mkdir -p $READS_DIR
    mkdir -p $TRIAL_OUTPUT_DIR

    OUTPUT_FILENAME="abundances.out"
    SEED=$trial

    # Generate the reads.
    python $PROJECT_DIR/scripts/simulate_reads.py \
    --seed $SEED \
    --out_dir $READS_DIR \
    --abundance_path $TRUE_ABUNDANCE_PATH \
    --num_reads $n_reads \
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
    --out_dir $TRIAL_OUTPUT_DIR \
    --abundances_file $OUTPUT_FILENAME \
    --skip_filter

    echo "\"Chronostrain\",${n_reads},\"${TRIAL_OUTPUT_DIR}/${OUTPUT_FILENAME}\"" >> $TRIALS_INDEX_PATH
  done
done


# Plot the result.
PERFORMANCE_PLOT_PATH="${OUTPUT_DIR}/performance_plot.${PLOT_FORMAT}"

python ${PROJECT_DIR}/scripts/plot_performances.py \
--trial_specification $TRIALS_INDEX_PATH \
--ground_truth_path $TRUE_ABUNDANCE_PATH \
--output_path $PERFORMANCE_PLOT_PATH \
--font_size 18 \
--thickness 3 \
--format "${PLOT_FORMAT}"