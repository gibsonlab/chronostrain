#!/bin/bash
set -e

# =====================================
# Change this to where project is located. Should be able to call `python scripts/run_inference.py`.
PROJECT_DIR="/mnt/f/microbiome_tracking"
# =====================================

# ======================================
# filesystem paths (relative to PROJECT_DIR) --> no need to modify.
BASE_DIR="${PROJECT_DIR}/examples/em_perf_sparse"

CHRONOSTRAIN_INI="${BASE_DIR}/chronostrain.ini"
CHRONOSTRAIN_LOG_INI="${BASE_DIR}/logging.ini"
CHRONOSTRAIN_LOG_FILEPATH="${BASE_DIR}/logs/em_perf_sparse.log"

READS_DIR="${BASE_DIR}/simulated_reads"
TRUE_ABUNDANCE_PATH_SAMPLING="${BASE_DIR}/true_abundances_biased.csv"
TRUE_ABUNDANCE_PATH_PLOTTING="${BASE_DIR}/true_abundances.csv"
OUTPUT_DIR_T_ON="${BASE_DIR}/output/time_correlation_on"
OUTPUT_DIR_T_OFF="${BASE_DIR}/output/time_correlation_off"
ABUNDANCES_OUT_FILENAME="abundances.out"

PLOT_FORMAT="pdf"
INFERENCE_METHOD="em"

depth=200
sparse_depth=50
# =====================================

export BASE_DIR
export CHRONOSTRAIN_INI
export CHRONOSTRAIN_LOG_INI
export CHRONOSTRAIN_LOG_FILEPATH
mkdir -p $OUTPUT_DIR_T_ON
mkdir -p $OUTPUT_DIR_T_OFF

# Sample reads.
python $PROJECT_DIR/scripts/simulate_reads.py \
--out_dir $READS_DIR \
--abundance_path "${TRUE_ABUNDANCE_PATH_SAMPLING}" \
--seed 123 \
--num_reads $depth $depth $sparse_depth $depth \
--read_length 150

# =============== Run inference algorithm.
# Time consistency on
python $PROJECT_DIR/scripts/run_inference.py \
--reads_dir $READS_DIR \
--true_abundance_path $TRUE_ABUNDANCE_PATH_PLOTTING \
--method $INFERENCE_METHOD \
--read_length 150 \
--seed 123 \
-lr 0.001 \
--iters 50000 \
--out_dir $OUTPUT_DIR_T_ON \
--plot_format $PLOT_FORMAT \
--abundances_file $ABUNDANCES_OUT_FILENAME


# Time consistency off
python $PROJECT_DIR/scripts/run_inference.py \
--reads_dir $READS_DIR \
--true_abundance_path $TRUE_ABUNDANCE_PATH_PLOTTING \
--method $INFERENCE_METHOD \
--read_length 150 \
--seed 123 \
-lr 0.001 \
--iters 50000 \
--out_dir $OUTPUT_DIR_T_OFF \
--plot_format $PLOT_FORMAT \
--abundances_file $ABUNDANCES_OUT_FILENAME \
--disable_time_consistency


# ============== Redraw plots (styled for paper.)
python scripts/plot_abundance_output.py \
--abundance_path "${OUTPUT_DIR_T_ON}/${ABUNDANCES_OUT_FILENAME}" \
--output_path "${OUTPUT_DIR_T_ON}/EM_biased_with_correction.${PLOT_FORMAT}" \
--ground_truth_path $TRUE_ABUNDANCE_PATH_PLOTTING \
--font_size 18 \
--thickness 3 \
--title "With correction" \
--ylim 0.0 0.7 \
--format $PLOT_FORMAT

python scripts/plot_abundance_output.py \
--abundance_path "${OUTPUT_DIR_T_OFF}/${ABUNDANCES_OUT_FILENAME}" \
--abundance_path "${OUTPUT_DIR_T_OFF}/EM_biased_without_correction.${PLOT_FORMAT}" \
--ground_truth_path $TRUE_ABUNDANCE_PATH_PLOTTING \
--font_size 18 \
--thickness 3 \
--title "Without correction" \
--ylim 0.0 0.7 \
--format $PLOT_FORMAT