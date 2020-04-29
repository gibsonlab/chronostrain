#!/bin/bash

cd ..
python simulate_reads.py \
-o "./data/simulated_data/test_1" \
-a "ncbi_refs_1.csv" \
-b "strain_abundances_1.csv" \
-n 150 \
-l 500 \
-s 123 \
-trim 2500

python run_inference.py \
-d "data/simulated_data/test_1" \
-r "sim_reads_t1.fastq" \
"sim_reads_t2.fastq" \
"sim_reads_t3.fastq" \
"sim_reads_t4.fastq" \
-a "ncbi_refs_1.csv" \
-t 1 2 3 4 \
-m "em" \
-y "output/test_1" \
-o "inferred_abundances_1.csv" \
-b "strain_abundances_1.csv" \
-s 123