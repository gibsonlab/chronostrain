#!/bin/bash
set -e

TESTNAME="em_perf_lowquality"
SPARSE_DEPTH="10"

echo "----------------------------------- Quality [ON], testname ${TESTNAME} -----------------------------------"
for depth in 10 30 50 70 90 100 200 300 400 500 600 700 800 900
do
  for trial in 1 2 3 4 5 6 7 8 9 10
  do
    python3 plot_abundance_output.py \
    --abundance_path "data/output/test_$TESTNAME/depth_$depth-trial_$trial/EM_result_$TESTNAME.csv" \
    --output_path "data/output/test_$TESTNAME/depth_$depth-trial_$trial/plot.png" \
    --ground_truth_path "tests/em_perf/true_abundances.csv" \
    --font_size 18 \
    --thickness 3
  done
done

TESTNAME_qoff="em_perf_argmax_lowquality"

echo "----------------------------------- Quality [OFF], testname ${TESTNAME_qoff} -----------------------------------"
for depth in 10 30 50 70 90 100 200 300 400 500 600 700 800 900
do
  for trial in 1 2 3 4 5 6 7 8 9 10
  do
    python3 plot_abundance_output.py \
    --abundance_path "data/output/test_${TESTNAME_qoff}/depth_$depth-trial_$trial/EM_result_${TESTNAME_qoff}.csv" \
    --output_path "data/output/test_${TESTNAME_qoff}/depth_$depth-trial_$trial/plot.png" \
    --ground_truth_path "tests/em_perf/true_abundances.csv" \
    --font_size 18 \
    --thickness 3
  done
done

echo "--------------------------------- Plotting comparisons. ----------------------------------------------"
trial_params=""
qoff_trial_params=""
for depth in 10 30 50 70 90 100 200 300 400 500 600 700 800 900
do
  for trial in 1 2 3 4 5 6 7 8 9 10
  do
    trial_params="$trial_params -t qon $depth data/output/test_$TESTNAME/depth_$depth-trial_$trial/EM_result_$TESTNAME.csv"
    qoff_trial_params="${qoff_trial_params} -t qoff $depth data/output/test_${TESTNAME_qoff}/depth_$depth-trial_$trial/EM_result_${TESTNAME_qoff}.csv"
  done
done

python3 plot_performances.py \
--ground_truth_path "tests/em_perf/true_abundances.csv" \
--output_path "data/output/test_$TESTNAME/performance_plot.png" \
--font_size 18 \
--thickness 3 \
--draw_legend \
$trial_params $qoff_trial_params
