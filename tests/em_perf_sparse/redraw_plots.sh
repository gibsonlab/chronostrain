#!/bin/bash
set -e

depth=500
trial=1

python3 plot_abundance_output.py \
--abundance_path "data/output/test_em_perf_sparse/depth_$depth-trial_$trial/EM_result_em_perf_sparse.csv" \
--output_path "data/output/test_em_perf_sparse/depth_$depth-trial_$trial/EM_biased_with_correction.png" \
--ground_truth_path "tests/em_perf_sparse/true_abundances.csv" \
--font_size 18 \
--thickness 3 \
--title "With correction" \
--ylim 0.0 0.7

python3 plot_abundance_output.py \
--abundance_path "data/output/test_em_perf_sparse/depth_$depth-trial_$trial-timeoff/EM_result_em_perf_sparse.csv" \
--output_path "data/output/test_em_perf_sparse/depth_$depth-trial_$trial-timeoff/EM_biased_without_correction.png" \
--ground_truth_path "tests/em_perf_sparse/true_abundances.csv" \
--font_size 18 \
--thickness 3 \
--title "Without correction" \
--ylim 0.0 0.7
