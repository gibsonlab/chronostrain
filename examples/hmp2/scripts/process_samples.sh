#!/bin/bash
set -e
source settings.sh "verbose"

export BASE_DIR
xargs -t -n 1 -P 1 'bash ${BASE_DIR}/helpers/process_patient.sh $1' < ${BASE_DIR}/files/patients.txt
