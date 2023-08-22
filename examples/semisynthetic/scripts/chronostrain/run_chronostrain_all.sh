#!/bin/bash
set -e
source settings.sh

cd ${BASE_DIR}/scripts

for (( trial = 1; trial < ${N_TRIALS}+1; trial++ )); do
	for n_reads in "${SYNTHETIC_COVERAGES[@]}"; do
		bash chronostrain/run_chronostrain.sh $n_reads $trial
	done
done