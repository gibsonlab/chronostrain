#!/bin/bash
set -e
source settings.sh


cd ${BASE_DIR}/scripts
for n_reads in 10000 50000 100000 500000 1000000
do
	for (( trial = 1; trial < ${N_TRIALS}+1; trial++ ));
	do
		bash filter.sh $n_reads $trial
	done
done