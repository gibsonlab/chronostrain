#!/bin/bash
set -e

TESTNAME="em_perf"
ACCESSION="tests/em_perf/ncbi_refs.csv"
SRC_ABUNDANCE="tests/em_perf/true_abundances.csv"
TRIM=500
ITERS=50000

echo "----------------------------------- testname ${TESTNAME} -----------------------------------"
for depth in 10 30 50 70 90 100 200 300 400 500 600 700 800 900
do
  for trial in 1 2 3 4 5 6 7 8 9 10
  do
    echo "------------------------------------ depth $depth, trial $trial -----------------------------"

    echo "------------------------------------- Sample Reads ----------------------------------"
    python3 simulate_reads.py \
		--out_dir "./data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/" \
		--accession_path "$ACCESSION" \
		--abundance_path "$SRC_ABUNDANCE" \
		--num_reads $depth $depth $sparse_depth $depth \
		--read_length 150 \
		-trim $TRIM

    echo "------------------------------------- Perform Inference ----------------------------------"
	  python3 run_inference.py \
		--read_files \
		"data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t1.fastq" \
		"data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t2.fastq" \
		"data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t3.fastq" \
		"data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t4.fastq" \
		"data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t5.fastq" \
		"data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t10.fastq" \
		"data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t11.fastq" \
		"data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t17.fastq" \
		--true_abundance_path "tests/$TESTNAME/true_abundances_renormalized.csv" \
		--accession_path "$ACCESSION" \
		--time_points 1 2 3 4 5 10 11 17 \
		--method "em" \
		--out_path "data/output/$TESTNAME/depth_$depth-trial_$trial/EM_result_$TESTNAME.csv" \
		--plots_path "data/output/$TESTNAME/depth_$depth-trial_$trial/EM_result_${TESTNAME}_plot.png" \
		-trim $TRIM \
		--iters $ITERS

    echo "------------------------------------- Redraw Plot (without legend/title) ----------------------------------"
    python3 plot_abundance_output.py \
    --abundance_path "data/output/$TESTNAME/depth_$depth-trial_$trial/EM_result_$TESTNAME.csv" \
    --output_path "data/output/$TESTNAME/depth_$depth-trial_$trial/plot.png" \
    --ground_truth_path "tests/$TESTNAME/true_abundances_renormalized.csv" \
    --font_size 18 \
    --thickness 3
  done
done

echo "--------------------------------- Plotting comparisons. ----------------------------------------------"
trial_params=""
for depth in 10 30 50 70 90 100 200 300 400 500 600 700 800 900
do
  for trial in 1 2 3 4 5 6 7 8 9 10
  do
    trial_params="$trial_params -t qon $depth data/output/$TESTNAME/depth_$depth-trial_$trial/EM_result_$TESTNAME.csv"
  done
done

python3 plot_performances.py \
--ground_truth_path "tests/em_perf/true_abundances_renormalized.csv" \
--output_path "data/output/$TESTNAME/performance_plot.png" \
--font_size 18 \
--thickness 3 \
$trial_params
