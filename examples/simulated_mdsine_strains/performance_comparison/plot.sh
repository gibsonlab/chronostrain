#!/bin/bash

# =====================================
# Change this to where project is located. Should be able to call `python scripts/run_inference.py`.
PROJECT_DIR="/mnt/f/microbiome_tracking"
# =====================================


# ======================================
# filesystem paths (relative to PROJECT_DIR) --> no need to modify.
BASE_DIR="${PROJECT_DIR}/examples/simulated_mdsine_strains/performance_comparison"

CHRONOSTRAIN_INI="${BASE_DIR}/chronostrain.ini"
CHRONOSTRAIN_LOG_INI="${BASE_DIR}/logging.ini"
CHRONOSTRAIN_LOG_FILEPATH="${BASE_DIR}/logs_${NUM_READS}reads/run.log"

TRUE_ABUNDANCE_PATH="${BASE_DIR}/true_abundances.csv"
OUTPUT_DIR="${BASE_DIR}/output"
TRIALS_INDEX_PATH="${OUTPUT_DIR}/trials_index.csv"
PERFORMANCE_PLOT_PATH="${OUTPUT_DIR}/performance_plot.png"
# =====================================

export BASE_DIR
export CHRONOSTRAIN_INI
export CHRONOSTRAIN_LOG_INI
export CHRONOSTRAIN_LOG_FILEPATH

# Generate the index file.
mkdir -p $OUTPUT_DIR
> $TRIALS_INDEX_PATH

TRIAL_BASEDIR="${OUTPUT_DIR}/trials"
dirnames=`ls ${TRIAL_BASEDIR}`
for trial_dirname in $dirnames
do
  # Extract n_tokens from directory name, of pattern "reads_<n_reads>_trial_<trial>
  # https://stackoverflow.com/questions/10586153/how-to-split-a-string-into-an-array-in-bash
  IFS='_' read -r -a tokens <<< "$trial_dirname"
  n_reads=${tokens[1]}

  echo "\"Chronostrain\" ${n_reads} \"${TRIAL_BASEDIR}/${trial_dirname}/output/abundances.out\"" >> $TRIALS_INDEX_PATH
done

# Plot the result.
python ${PROJECT_DIR}/scripts/plot_performances.py \
--trial_specification $TRIALS_INDEX_PATH \
--ground_truth_path $TRUE_ABUNDANCE_PATH \
--output_path $PERFORMANCE_PLOT_PATH \
--font_size 18 \
--thickness 3