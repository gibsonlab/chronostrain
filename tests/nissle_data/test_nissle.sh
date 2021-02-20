#!/bin/bash
set -e

TESTNAME="nissle_data"

# Time consistency off
python run_inference.py \
--base_path "data/simulated_reads/$TESTNAME/" \
--read_files "all_reads.fq" \
--true_abundance_path "data/simulated_reads/$TESTNAME/sim_abundances.csv" \
--accession_path "tests/$TESTNAME/ncbi_refs.json" \
--time_points 1 \
--method "em" \
--read_length 150 \
--seed 123 \
--out_path "data/output/test_${TESTNAME}_time_off/EM_result_${TESTNAME}_time_off.csv" \
--plots_path "data/output/test_${TESTNAME}_time_off/EM_result_${TESTNAME}_plot_time_off.png" \
--disable_time_consistency