#!/bin/bash
set -e

TESTNAME="em_perf_sparse"
depth=500
for sparse_depth in 10 50 100 150 200 250 300 350 400 450 500 550 600 650 600 650 700
do
  for trial in 1 2 3 4 5 6 7 8 9 10
    do
	python3 simulate_reads.py \
		--out_dir "./data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/" \
		--accession_path "tests/$TESTNAME/ncbi_refs.csv" \
		--abundance_path "tests/$TESTNAME/true_abundances_2.csv" \
		--num_reads $depth $depth $sparse_depth $depth \
		--read_length 150 \
		-trim 500
	# Time consistency on
	python3 run_inference.py \
		--read_files \
		"data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t1.fastq" \
		"data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t2.fastq" \
		"data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t3.fastq" \
		"data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t4.fastq" \
		--true_abundance_path "data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_abundances.csv" \
		--accession_path "tests/$TESTNAME/ncbi_refs.csv" \
		--time_points 1 2 3 4 \
		--method "em" \
		--out_path "data/output/test_$TESTNAME/depth_$depth-trial_$trial/EM_result_$TESTNAME.csv" \
		--plots_path "data/output/test_$TESTNAME/depth_$depth-trial_$trial/EM_result_${TESTNAME}_plot.png" \
		-trim 500 \
		--iters 50000
	# Time consistency off
	python3 run_inference.py \
		--read_files \
		"data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t1.fastq" \
		"data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t2.fastq" \
		"data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t3.fastq" \
		"data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t4.fastq" \
		--true_abundance_path "data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_abundances.csv" \
		--accession_path "tests/$TESTNAME/ncbi_refs.csv" \
		--time_points 1 2 3 4 \
		--method "em" \
		--out_path "data/output/test_$TESTNAME/depth_$depth-trial_$trial-timeoff/EM_result_$TESTNAME.csv" \
		--plots_path "data/output/test_$TESTNAME/depth_$depth-trial_$trial-timeoff/EM_result_${TESTNAME}_plot.png" \
		-trim 500 \
		--iters 50000 \
		--disable_time_consistency
    done
done
