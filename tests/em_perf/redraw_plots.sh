#!/bin/bash
set -e

TESTNAME="em_perf"
SPARSE_DEPTH="10"

for depth in 10 100 200 300 400 500 600 700 800 900 1000 1500 2000 3000
do
  for trial in 1 2 3 4 5 6 7 8 9 10
  do
    python plot_abundance_output.py \
    --abundance_path "data/output/test_$TESTNAME/depth_$depth-trial_$trial/EM_result_$TESTNAME.csv" \
    --output_path "data/output/test_$TESTNAME/depth_$depth-trial_$trial/EM_result_${TESTNAME}_plot2.png" \
    --num_reads $depth $depth $depth $SPARSE_DEPTH $depth $depth $SPARSE_DEPTH $depth \
    --ground_truth_path "data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_abundances.csv"
  done
done

trial_params=""
for depth in 10 100 200 300 400 500 600 700 800 900 1000 1500 2000 3000
do
  for trial in 1 2 3 4 5 6 7 8 9 10
  do
    trial_params="$trial_params -t $depth data/output/test_$TESTNAME/depth_$depth-trial_$trial/EM_result_$TESTNAME.csv"
  done
done

python plot_performances \
--ground_truth_path "data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_abundances.csv" \
--output_path "data/output/test_$TESTNAME/performance_plot.png" \
$trial_params