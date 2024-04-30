#!/bin/bash
set -e
source settings.sh

for (( trial = 1; trial < ${N_TRIALS}+1; trial++ )); do
	for n_reads in 10000 25000 50000 75000 100000; do
		bash strainfacts/run_gtpro.sh $n_reads $trial
	done
done