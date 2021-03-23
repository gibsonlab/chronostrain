#!/bin/bash
set -e

TESTNAME="nissle_data"
cd ../..

# Time consistency off
python scripts/run_inference.py \
--reads_dir "data/simulated_reads/$TESTNAME/" \
--read_files "data/$TESTNAME/all_reads.fq" \
--true_abundance_path "data/$TESTNAME/sim_abundances.csv" \
--method "em" \
--read_length 150 \
--seed 123 \
--out_dir "data/output/${TESTNAME}" \
--disable_time_consistency
