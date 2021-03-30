#!/bin/bash
set -e

# =====================================
# Change this to where project is located. Should be able to call `python scripts/run_inference.py`.
PROJECT_DIR="/mnt/f/microbiome_tracking"
# =====================================

# ======================================
# filesystem paths (relative to PROJECT_DIR) --> no need to modify.
BASE_DIR="${PROJECT_DIR}/examples/adversarial_test_icml"

CHRONOSTRAIN_INI="${BASE_DIR}/chronostrain.ini"
CHRONOSTRAIN_LOG_INI="${BASE_DIR}/logging.ini"
CHRONOSTRAIN_LOG_FILEPATH="${BASE_DIR}/logs/adversarial_icml.log"

READS_BASE_DIR="${BASE_DIR}/reads"
TRUE_ABUNDANCE_PATH="${BASE_DIR}/true_abundances.csv"
OUTPUT_DIR="${BASE_DIR}/output"
TRIALS_INDEX_PATH="${OUTPUT_DIR}/trials_index.csv"

PLOT_FORMAT="pdf"
INFERENCE_METHOD="em"
SEED=123
# =====================================

export BASE_DIR
export CHRONOSTRAIN_INI
export CHRONOSTRAIN_LOG_INI
export CHRONOSTRAIN_LOG_FILEPATH
export USE_QUALITY="True"

mkdir -p $OUTPUT_DIR

# Generate reads (deterministic/adversarial)
python $BASE_DIR/scripts/create_reads.py \
--reads_dir $READS_BASE_DIR

# Erase contents of trials.csv
> $TRIALS_INDEX_PATH

# Run inference.
for depth in 20 40 60 80 100 120 140 160 180 200
do
  READS_DIR="${READS_BASE_DIR}/depth_${depth}"
  OUTPUT_FILENAME_QON="qon_depth_${depth}.out"
  OUTPUT_FILENAME_QOFF="qoff_depth_${depth}.out"

  # Run chronostrain, quality scores used in the model.
  export USE_QUALITY="True"
  echo "$PROJECT_DIR/scripts/run_inference.py --reads_dir $READS_DIR --true_abundance_path $TRUE_ABUNDANCE_PATH --method $INFERENCE_METHOD --read_length 50 --seed $SEED -lr 0.001 --iters 10000 --out_dir $OUTPUT_DIR --abundances_file $OUTPUT_FILENAME_QON --skip_filter"
  python $PROJECT_DIR/scripts/run_inference.py \
  --reads_dir $READS_DIR \
  --true_abundance_path $TRUE_ABUNDANCE_PATH \
  --method $INFERENCE_METHOD \
  --read_length 50 \
  --seed $SEED \
  -lr 0.001 \
  --iters 10000 \
  --out_dir $OUTPUT_DIR \
  --abundances_file $OUTPUT_FILENAME_QON \
  --skip_filter

  # Run chronostrain, quality scores turned off.
  export USE_QUALITY="False"
  echo "$PROJECT_DIR/scripts/run_inference.py --reads_dir $READS_DIR --true_abundance_path $TRUE_ABUNDANCE_PATH --method $INFERENCE_METHOD --read_length 50 --seed $SEED -lr 0.001 --iters 10000 --out_dir $OUTPUT_DIR --abundances_file $OUTPUT_FILENAME_QOFF --skip_filter"
  python $PROJECT_DIR/scripts/run_inference.py \
  --reads_dir $READS_DIR \
  --true_abundance_path $TRUE_ABUNDANCE_PATH \
  --method $INFERENCE_METHOD \
  --read_length 50 \
  --seed $SEED \
  -lr 0.001 \
  --iters 10000 \
  --out_dir $OUTPUT_DIR \
  --abundances_file $OUTPUT_FILENAME_QOFF \
  --skip_filter

  echo "\"Full error model\",${depth},\"${OUTPUT_DIR}/${OUTPUT_FILENAME_QON}\"" >> $TRIALS_INDEX_PATH
  echo "\"No error model\",${depth},\"${OUTPUT_DIR}/${OUTPUT_FILENAME_QOFF}\"" >> $TRIALS_INDEX_PATH
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
