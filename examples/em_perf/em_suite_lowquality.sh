#!/bin/bash
set -e

TESTNAME="em_perf_lowquality"
TESTNAME_qoff="em_perf_argmax_lowquality"
SPARSE_DEPTH="10"
ACCESSION="examples/em_perf/ncbi_refs.csv"
SRC_ABUNDANCE="examples/em_perf/true_abundances.csv"
TRIM=500
ITERS=50000
cd ../..

echo "----------------------------------- testname ${TESTNAME} -----------------------------------"

for depth in 10 30 50 70 90 100 200 300 400 500 600 700 800 900
do
  for trial in 1 2 3 4 5 6 7 8 9 10
  do
    echo "------------------------------------ Sample Reads (depth $depth, trial $trial) ------------------------------"
    python3 scripts/simulate_reads.py \
		--out_dir "./data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/" \
		--accession_path "$ACCESSION" \
		--abundance_path "$SRC_ABUNDANCE" \
		--num_reads $depth $depth $sparse_depth $depth \
		--read_length 150 \
		-trim $TRIM
	done
done


echo "----------------------------------- Quality [ON], testname ${TESTNAME} -----------------------------------"
for depth in 10 30 50 70 90 100 200 300 400 500 600 700 800 900
do
  for trial in 1 2 3 4 5 6 7 8 9 10
  do
    echo "-------------------------------- Inference (Quality on, depth $depth, trial $trial) ------------------------"
	  python3 scripts/run_inference.py \
		--read_files \
		"data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t1.fastq" \
		"data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t2.fastq" \
		"data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t3.fastq" \
		"data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t4.fastq" \
		"data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t5.fastq" \
		"data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t10.fastq" \
		"data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t11.fastq" \
		"data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t17.fastq" \
		--true_abundance_path "data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_abundances.csv" \
		--accession_path "$ACCESSION" \
		--time_points 1 2 3 4 5 10 11 17 \
		--method "em" \
		--out_path "data/output/$TESTNAME/depth_$depth-trial_$trial/EM_result_$TESTNAME.csv" \
		--plots_path "data/output/$TESTNAME/depth_$depth-trial_$trial/EM_result_${TESTNAME}_plot.png" \
		-trim $TRIM \
		--iters $ITERS

    python3 scripts/plot_abundance_output.py \
    --abundance_path "data/output/$TESTNAME/depth_$depth-trial_$trial/EM_result_$TESTNAME.csv" \
    --output_path "data/output/$TESTNAME/depth_$depth-trial_$trial/plot.png" \
    --ground_truth_path "examples/em_perf/true_abundances.csv" \
    --font_size 18 \
    --thickness 3
  done
done


echo "----------------------------------- Quality [OFF], testname ${TESTNAME_qoff} -----------------------------------"
for depth in 10 30 50 70 90 100 200 300 400 500 600 700 800 900
do
  for trial in 1 2 3 4 5 6 7 8 9 10
  do
    echo "------------------------------- Inference (Quality off, depth $depth, trial $trial) ------------------------"
    python3 scripts/run_inference.py \
		--read_files \
		"data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t1.fastq" \
		"data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t2.fastq" \
		"data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t3.fastq" \
		"data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t4.fastq" \
		"data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t5.fastq" \
		"data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t10.fastq" \
		"data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t11.fastq" \
		"data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t17.fastq" \
		--true_abundance_path "data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_abundances.csv" \
		--accession_path "$ACCESSION" \
		--time_points 1 2 3 4 5 10 11 17 \
		--method "em" \
		--out_path "data/output/$TESTNAME/depth_$depth-trial_$trial/EM_result_$TESTNAME.csv" \
		--plots_path "data/output/$TESTNAME/depth_$depth-trial_$trial/EM_result_${TESTNAME}_plot.png" \
		-trim $TRIM \
		--iters $ITERS \
		--disable_quality

    python3 scripts/plot_abundance_output.py \
    --abundance_path "data/output/test_${TESTNAME_qoff}/depth_$depth-trial_$trial/EM_result_${TESTNAME_qoff}.csv" \
    --output_path "data/output/test_${TESTNAME_qoff}/depth_$depth-trial_$trial/plot.png" \
    --ground_truth_path "examples/em_perf/true_abundances.csv" \
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
    trial_params="$trial_params -t qon $depth data/output/$TESTNAME/depth_$depth-trial_$trial/EM_result_$TESTNAME.csv"
    qoff_trial_params="${qoff_trial_params} -t qoff $depth data/output/test_${TESTNAME_qoff}/depth_$depth-trial_$trial/EM_result_${TESTNAME_qoff}.csv"
  done
done

python3 scripts/plot_performances.py \
--ground_truth_path "examples/em_perf/true_abundances.csv" \
--output_path "data/output/$TESTNAME/performance_plot.png" \
--font_size 18 \
--thickness 3 \
--draw_legend \
$trial_params $qoff_trial_params
