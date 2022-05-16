#!/bin/bash
set -e
source settings.sh


export CHRONOSTRAIN_CACHE_DIR=.
python ${BASE_DIR}/helpers/evaluate.py \
-i ${REFSEQ_INDEX} \
-b ${DATA_DIR} \
-o ${DATA_DIR}/summary/output.csv \
-a /mnt/d/chronostrain/umb_database/strain_alignments/concatenation.fasta
