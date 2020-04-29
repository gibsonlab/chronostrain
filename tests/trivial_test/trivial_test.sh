#!/bin/bash

cd ..
python simulate_reads.py \
-s 123 \
-o "./data/simulated_data/trivial_test" \
-a "ncbi_refs_2.csv" \
-b "strain_abundances_2.csv" \
-n 1  \
-l 150 \
-trim 200

python run_inference.py \
-d "data/simulated_data/trivial_test" \
-r "sim_reads_t1.fastq" \
"sim_reads_t2.fastq" \
"sim_reads_t4.fastq" \
"sim_reads_t6.fastq" \
"sim_reads_t7.fastq" \
"sim_reads_t8.fastq" \
"sim_reads_t10.fastq" \
-a "ncbi_refs_2.csv" \
-t 1 2 4 6 7 8 10 \
-m "em" \
-y "data/inference_output/trivial_test" \
-o "inferred_abundances_0.csv" \
-b "strain_abundances_2.csv" \
-s 123 \
-trim 200