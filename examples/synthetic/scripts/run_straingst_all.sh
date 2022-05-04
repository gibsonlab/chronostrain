#!/bin/bash
set -e
source settings.sh


cd ${BASE_DIR}/scripts
bash create_straingst_db.sh

for n_reads in 10000 50000 100000 500000 1000000; do
	for (( trial = 1; trial < ${N_TRIALS}+1; trial++ )); do
		for time_point in 0 1 2 3 4; do
			bash run_straingst.sh $n_reads $trial $time_point
		done
	done
done
