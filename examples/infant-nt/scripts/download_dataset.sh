#!/bin/bash
set -e
source settings.sh

participant=$1
require_variable 'participant' $participant
require_file $DATASET_METAGENOMIC_CATALOG


python ${BASE_DIR}/helpers/dataset_download.py \
  -m ${DATASET_METAGENOMIC_CATALOG} \
  -o ${DATA_DIR} \
  -p ${participant}


cd ${BASE_DIR}/scripts
bash ../helpers/process_dataset.sh ${participant}  # Run pre-processing on reads.
