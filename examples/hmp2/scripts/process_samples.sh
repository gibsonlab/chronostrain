#!/bin/bash
set -e
source settings.sh

xargs -t -n 1 -P 1 ${BASE_DIR}/helpers/process_patient.sh < ${BASE_DIR}/files/patients.txt
