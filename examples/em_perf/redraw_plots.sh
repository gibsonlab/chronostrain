#!/bin/bash
set -e

TESTNAME="em_perf"
cd ../..

#for depth in 10 100 200 300 400 500 600 700 800 900
#for depth in 500
#do
#  for trial in 1 2 3 4 5
#  do
#    python3 plot_abundance_output.py \
#    --abundance_path "data/output/$TESTNAME/depth_$depth-trial_$trial/EM_result_$TESTNAME.csv" \
#    --output_path "data/output/$TESTNAME/depth_$depth-trial_$trial/plot.png" \
#    --ground_truth_path "examples/em_perf/true_abundances.csv" \
#    --font_size 18 \
#    --thickness 3
#  done
#done

trial_params=""
for depth in 5 10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800 900
do
  for trial in 1 2 3 4 5 6 7 8 9 10
  do
    trial_params="$trial_params -t em_perf $depth data/output/$TESTNAME/depth_$depth-trial_$trial/EM_result_$TESTNAME.csv"
  done
done

echo $trial_params

python3 scripts/plot_performances.py \
--ground_truth_path "examples/em_perf/true_abundances.csv" \
--output_path "data/output/$TESTNAME/performance_plot.png" \
--font_size 18 \
--thickness 3 \
$trial_params
