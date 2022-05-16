#!/bin/bash
set -e
source settings.sh


python ${BASE_DIR}/helpers/evaluate.py \
-i ${REFSEQ_INDEX} \
-a /mnt/d/chronostrain/umb_database/strain_alignments/concatenation.fasta
