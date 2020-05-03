#!/bin/bash

TESTNAME="8strains"

cd ..
python simulate_reads.py \
-s 123 \
--out_dir "./data/simulated_reads/$TESTNAME" \
--accession_path "tests/$TESTNAME/ncbi_refs.csv" \
--abundance_path "tests/$TESTNAME/strain_abundances.csv" \
--num_reads 500 \
--read_lengths 150 \
-trim 2500

python run_inference.py \
--read_files \
"data/simulated_reads/$TESTNAME/sim_reads_t1.fastq" \
"data/simulated_reads/$TESTNAME/sim_reads_t2.fastq" \
"data/simulated_reads/$TESTNAME/sim_reads_t3.fastq" \
"data/simulated_reads/$TESTNAME/sim_reads_t4.fastq" \
--accession_path "tests/$TESTNAME/ncbi_refs.csv" \
--true_abundance_path "tests/$TESTNAME/strain_abundances.csv" \
--time_points 1 2 3 4 \
--method "em" \
--seed 123 \
--out_dir "data/output/test_$TESTNAME" \
--out_file "EM_result_$TESTNAME.csv" \
--plots_file "EM_result_$TESTNAME_plot.png" \
-trim 2500 \
--time_consistency "on"


python run_inference.py \
--read_files \
"data/simulated_reads/$TESTNAME/sim_reads_t1.fastq" \
"data/simulated_reads/$TESTNAME/sim_reads_t2.fastq" \
"data/simulated_reads/$TESTNAME/sim_reads_t3.fastq" \
"data/simulated_reads/$TESTNAME/sim_reads_t4.fastq" \
--accession_path "tests/$TESTNAME/ncbi_refs.csv" \
--true_abundance_path "tests/$TESTNAME/strain_abundances.csv" \
--time_points 1 2 3 4 \
--method "em" \
--seed 123 \
--out_dir "data/output/test_${TESTNAME_SUFFIX}_time_off" \
--out_file "EM_result_${TESTNAME_SUFFIX}_time_off.csv" \
--plots_file "EM_result_${TESTNAME_SUFFIX}_plot_time_off.png" \
-trim 2500 \
--time_consistency "off"

