#!/bin/bash
set -e

TESTNAME="2strains"

cd ../..
python scripts/simulate_reads.py \
--seed 123 \
--out_dir "./data/simulated_reads/$TESTNAME" \
--accession_path "examples/$TESTNAME/ncbi_refs.csv" \
--abundance_path "examples/$TESTNAME/strain_abundances.csv" \
--num_reads 500 \
--read_length 150 \
-trim 2500

# Time consistency on
python scripts/run_inference.py \
--read_files \
"data/simulated_reads/$TESTNAME/sim_reads_t1.fastq" \
"data/simulated_reads/$TESTNAME/sim_reads_t2.fastq" \
"data/simulated_reads/$TESTNAME/sim_reads_t4.fastq" \
"data/simulated_reads/$TESTNAME/sim_reads_t6.fastq" \
"data/simulated_reads/$TESTNAME/sim_reads_t7.fastq" \
"data/simulated_reads/$TESTNAME/sim_reads_t8.fastq" \
"data/simulated_reads/$TESTNAME/sim_reads_t10.fastq" \
--true_abundance_path "data/simulated_reads/$TESTNAME/sim_abundances.csv" \
--accession_path "examples/$TESTNAME/ncbi_refs.csv" \
--time_points 1 2 4 6 7 8 10 \
--method "em" \
--seed 123 \
--out_path "data/output/$TESTNAME/EM_result_$TESTNAME.csv" \
--plots_path "data/output/$TESTNAME/EM_result_${TESTNAME}_plot.png" \
-trim 2500

# Time consistency off
python scripts/run_inference.py \
--read_files \
"data/simulated_reads/$TESTNAME/sim_reads_t1.fastq" \
"data/simulated_reads/$TESTNAME/sim_reads_t2.fastq" \
"data/simulated_reads/$TESTNAME/sim_reads_t4.fastq" \
"data/simulated_reads/$TESTNAME/sim_reads_t6.fastq" \
"data/simulated_reads/$TESTNAME/sim_reads_t7.fastq" \
"data/simulated_reads/$TESTNAME/sim_reads_t8.fastq" \
"data/simulated_reads/$TESTNAME/sim_reads_t10.fastq" \
--true_abundance_path "data/simulated_reads/$TESTNAME/sim_abundances.csv" \
--accession_path "examples/$TESTNAME/ncbi_refs.csv" \
--time_points 1 2 4 6 7 8 10 \
--method "em" \
--seed 123 \
--out_path "data/output/test_${TESTNAME}_time_off/EM_result_${TESTNAME}_time_off.csv" \
--plots_path "data/output/test_${TESTNAME}_time_off/EM_result_${TESTNAME}_plot_time_off.png" \
-trim 2500 \
--disable_time_consistency