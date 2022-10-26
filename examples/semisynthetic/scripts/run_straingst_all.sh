#!/bin/bash
set -e
source settings.sh


mode=$1
echo "[*] Running StrainGST batch on mode \"${mode}\""

cd ${BASE_DIR}/scripts
for (( trial = 1; trial < ${N_TRIALS}+1; trial++ )); do
	for n_reads in "${SYNTHETIC_COVERAGES[@]}"; do
		bash run_straingst.sh $n_reads $trial 0 $mode
		bash run_straingst.sh $n_reads $trial 1 $mode
		bash run_straingst.sh $n_reads $trial 2 $mode
		bash run_straingst.sh $n_reads $trial 3 $mode
		bash run_straingst.sh $n_reads $trial 4 $mode
	done
done
