#!/bin/bash
set -e
source settings.sh

pass=$1

cd ${BASE_DIR}/scripts/read_depths
for n_reads in 10000 50000 100000 500000 1000000
do
	for (( trial = 1; trial < ${N_TRIALS}+1; trial++ ));
	do
		bash run_chronostrain.sh $n_reads $trial $pass
	done
done