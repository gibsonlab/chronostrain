#!/bin/bash
set -e

TESTNAME="nissle_data"
cd ../..

# Time consistency off
python scripts/run_inference.py \
--reads_dir "data/simulated_reads/$TESTNAME/" \
--read_files "all_reads.fq" \
--true_abundance_path "data/simulated_reads/$TESTNAME/sim_abundances.csv" \
--method "em" \
--read_length 150 \
--seed 123 \
--out_dir "data/output/test_${TESTNAME}" \
--disable_time_consistency



--reads_dir
"../data/simulated_reads/em_perf"
--true_abundance_path
"../data/simulated_reads/em_perf/sim_abundances.csv"
--method
"em"
--seed
123
-lr
0.001
--iters
3000
--read_length
150
--out_dir
"data/output/em_perf"
--skip_filter