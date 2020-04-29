#!/bin/bash

cd ..
python simulate_reads.py \
-s 123 \
-o "./simulated_data/test_2" \
-a "ncbi_refs_2.csv" \
-b "strain_abundances_2.csv" \
-n 150 \
-l 500 \
-trim 2500

python run_inference.py \
-d "simulated_data/test_2" \
-a "ncbi_refs_2.csv" \
-r \
"sim_reads_t1.fastq" \
"sim_reads_t2.fastq" \
"sim_reads_t4.fastq" \
"sim_reads_t6.fastq" \
"sim_reads_t7.fastq" \
"sim_reads_t8.fastq" \
"sim_reads_t10.fastq" \
-t 1 2 4 6 7 8 10 \
-m "em" \
-y "output/test_2" \
-o "inferred_abundances_2.csv" \
-b "strain_abundances_2.csv" \
-s 123