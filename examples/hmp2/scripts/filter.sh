#!/bin/bash
set -e

source settings.sh
SEED=31415

# =========== Read filtering. ===============
while IFS=, read -r patient
do
	echo "[*] Filtering reads for ${patient}"
	export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/${patient}_filter.log"
	chronostrain filter \
	--reads "${READS_DIR}/${patient}/${INPUT_INDEX_FILENAME}" \
	-o "${READS_DIR}/${patient}/filtered" \
	--identity-threshold 0.975
done < ${BASE_DIR}/files/patients.txt
