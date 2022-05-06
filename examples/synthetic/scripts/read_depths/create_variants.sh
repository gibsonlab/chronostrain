#!/bin/bash
set -e
source settings.sh


python ${BASE_DIR}/helpers/create_variants.py \
-i ${BASE_DIR}/files/variants.json \
-o ${CHRONOSTRAIN_DB_DIR}/assemblies
