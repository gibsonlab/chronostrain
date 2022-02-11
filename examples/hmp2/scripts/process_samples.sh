#!/bin/bash
set -e
source settings.sh "verbose"

export BASE_DIR
xargs -t -n 1 -P 1 sh -c ${BASE_DIR}/helpers/process_patient.sh < ${BASE_DIR}/files/patients.txt
