#!/bin/bash

python run_inference.py -d "simulated_data" \
-r "sampled_read_t1.fastq" "sampled_read_t2.fastq" "sampled_read_t4.fastq" "sampled_read_t5.fastq" \
-a "ncbi_refs.csv" \
-t 1 2 4 5 \
-l 25 \
-m "em" \
-s 123
