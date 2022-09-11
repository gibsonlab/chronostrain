#!/bin/bash
set -e
source settings.sh


sensitivity=$1
cd ${BASE_DIR}/scripts

for (( trial = 1; trial < ${N_TRIALS}+1; trial++ )); do
	for n_reads in "${SYNTHETIC_COVERAGES[@]}"; do
		bash run_strainest.sh $n_reads $trial 0 $sensitivity
		bash run_strainest.sh $n_reads $trial 1 $sensitivity
		bash run_strainest.sh $n_reads $trial 2 $sensitivity
		bash run_strainest.sh $n_reads $trial 3 $sensitivity
		bash run_strainest.sh $n_reads $trial 4 $sensitivity
	done
done