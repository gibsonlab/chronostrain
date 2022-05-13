#!/bin/bash
set -e
source settings.sh


cd ${BASE_DIR}/scripts/read_depths
for n_reads in 10000 50000 100000 500000 1000000
do
	for (( trial = 1; trial < ${N_TRIALS}+1; trial++ ));
	do
		bash run_strainest.sh $n_reads $trial 0
		bash run_strainest.sh $n_reads $trial 1
		bash run_strainest.sh $n_reads $trial 2
		bash run_strainest.sh $n_reads $trial 3
		bash run_strainest.sh $n_reads $trial 4
	done
done