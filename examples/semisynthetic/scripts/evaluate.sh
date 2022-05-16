#!/bin/bash
set -e
source settings.sh


export CHRONOSTRAIN_CACHE_DIR=.
python ${BASE_DIR}/helpers/evaluate.py \
-b ${DATA_DIR} \
-o ${DATA_DIR}/summary/output.csv \
-g ${RELATIVE_GROUND_TRUTH} \
-a /mnt/d/chronostrain/umb_database/strain_alignments/concatenation.fasta
