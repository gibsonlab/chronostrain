#!/bin/bash

python read_simulate.py -s 123 \
-o "./simulated_data" \
-a "ncbi_refs.csv" \
-b "strain_abundances.csv" \
-t 1 2 4 5 \
-n 10 5 9 8 \
-l 30