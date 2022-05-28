#!/bin/bash
set -e
source settings.sh


cd ${BASE_DIR}/scripts

for (( trial = 1; trial < ${N_TRIALS}+1; trial++ )); do
	for n_reads in 5000 10000 25000 50000 75000 100000; do
		bash run_strainest.sh $n_reads $trial 0 &
		bash run_strainest.sh $n_reads $trial 1 &
		bash run_strainest.sh $n_reads $trial 2 &
		bash run_strainest.sh $n_reads $trial 3 &
		bash run_strainest.sh $n_reads $trial 4 &
		wait
	done
done