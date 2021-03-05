#!/bin/bash
set -e

# Represents the planted sparse case, where the reads were sampled from a different abundance profile
# than the ground truth (Due to bias).
# See Figure 2 of paper.

TESTNAME="em_perf_sparse"
depth=500
sparse_depth=500
trial=1

python3 simulate_reads.py \
--out_dir "./data/simulated_reads/$TESTNAME/depth_$sparse_depth-trial_$trial/" \
--accession_path "examples/$TESTNAME/ncbi_refs.csv" \
--abundance_path "examples/$TESTNAME/true_abundances_biased.csv" \
--num_reads $depth $sparse_depth $depth \
--read_length 150 \
-trim 150 \
--seed $sparse_depth$trial

# Time consistency on
python3 run_inference.py \
--read_files \
"data/simulated_reads/$TESTNAME/depth_$sparse_depth-trial_$trial/sim_reads_t1_0.fastq" \
"data/simulated_reads/$TESTNAME/depth_$sparse_depth-trial_$trial/sim_reads_t1_1.fastq" \
"data/simulated_reads/$TESTNAME/depth_$sparse_depth-trial_$trial/sim_reads_t1_2.fastq" \
--true_abundance_path "examples/$TESTNAME/true_abundances.csv" \
--accession_path "examples/$TESTNAME/ncbi_refs.csv" \
--time_points 1.0 1.1 1.2 \
--method "em" \
-trim 150 \
--out_path "data/output/$TESTNAME/depth_$sparse_depth-trial_$trial/EM_result_$TESTNAME.csv" \
--plots_path "data/output/$TESTNAME/depth_$sparse_depth-trial_$trial/EM_result_${TESTNAME}_plot.png" \
--iters 20000

# Time consistency off
python3 run_inference.py \
--read_files \
"data/simulated_reads/$TESTNAME/depth_$sparse_depth-trial_$trial/sim_reads_t1_0.fastq" \
"data/simulated_reads/$TESTNAME/depth_$sparse_depth-trial_$trial/sim_reads_t1_1.fastq" \
"data/simulated_reads/$TESTNAME/depth_$sparse_depth-trial_$trial/sim_reads_t1_2.fastq" \
--true_abundance_path "examples/$TESTNAME/true_abundances.csv" \
--accession_path "examples/$TESTNAME/ncbi_refs.csv" \
--time_points 1.0 1.1 1.2 \
--method "em" \
-trim 150 \
--out_path "data/output/$TESTNAME/depth_$sparse_depth-trial_$trial-timeoff/EM_result_$TESTNAME.csv" \
--plots_path "data/output/$TESTNAME/depth_$sparse_depth-trial_$trial-timeoff/EM_result_${TESTNAME}_plot.png" \
--iters 20000 \
--disable_time_consistency
