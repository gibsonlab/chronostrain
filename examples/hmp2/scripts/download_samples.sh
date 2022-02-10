#!/bin/bash
source settings.sh "verbose"

if ! [[ -f $HMP2_CSV_PATH ]]; then
	echo "[*] Downloading HMP2 metadata file."
	mkdir -p ${SAMPLES_DIR}
	echo "${HMP2_CSV_PATH}"
	curl -o ${HMP2_CSV_PATH} "https://ibdmdb.org/tunnel/products/HMP2/Metadata/hmp2_metadata.csv"
fi

xargs -t -n 1 -P 1 ${BASE_DIR}/helpers/download_patient.sh < ${BASE_DIR}/files/patients.txt
