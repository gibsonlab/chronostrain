#!/bin/bash
set -e
source settings.sh

for n_reads in 5000 10000 25000 50000 75000 100000
do
	for (( trial = 1; trial < ${N_TRIALS}+1; trial++ ));
	do
		bash run_gtpro.sh $n_reads $trial
	done
done