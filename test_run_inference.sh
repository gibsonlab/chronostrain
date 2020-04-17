#!/bin/bash

python run_inference.py -d "simulated_data" \
-r "sampled_read_t1.fastq" "sampled_read_t2.fastq" "sampled_read_t4.fastq" "sampled_read_t5.fastq" \
-a "ncbi_refs2.csv" \
-t 1 2 4 5 \
-l 500 \
-m "em" \
-b "strain_abundances2.csv" \
-s 123
