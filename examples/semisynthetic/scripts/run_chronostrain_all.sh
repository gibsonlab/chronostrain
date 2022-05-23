#!/bin/bash
set -e
source settings.sh

pass=$1
cd ${BASE_DIR}/scripts

python ${BASE_DIR}/helpers/init_db.py \
-i "/mnt/d/chronostrain/umb_database/database_pruned.json" \
-o ${CHRONOSTRAIN_DB_JSON}

for n_reads in 10000 25000 50000 75000 100000
do
	for (( trial = 1; trial < ${N_TRIALS}+1; trial++ ));
	do
		bash run_chronostrain.sh $n_reads $trial $pass
	done
done