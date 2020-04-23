#!/bin/bash

python simulate_reads.py -s 123 \
-o "./simulated_data" \
-a "ncbi_refs2.csv" \
-b "strain_abundances2.csv" \
-t 1 2 4 5 \
-n 500 500 500 500 \
-l 500

#python simulate_reads.py -s 123 \
#-o "./simulated_data" \
#-a "ncbi_refs.csv" \
#-t 1 2 4 5 \
#-n 10 5 9 8 \
#-l 25