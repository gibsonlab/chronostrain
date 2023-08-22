#!/bin/bash
set -e
source settings.sh

cd ${BASE_DIR}/scripts

for (( trial = 1; trial < ${N_TRIALS}+1; trial++ )); do
	for n_reads in "${SYNTHETIC_COVERAGES[@]}"; do
		bash msweep/run_msweep.sh $n_reads $trial 0
		bash msweep/run_msweep.sh $n_reads $trial 1
		bash msweep/run_msweep.sh $n_reads $trial 2
		bash msweep/run_msweep.sh $n_reads $trial 3
		bash msweep/run_msweep.sh $n_reads $trial 4
	done
done
