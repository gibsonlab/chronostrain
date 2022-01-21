#!/bin/bash
set -e
source ../settings.sh
export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/concatenate_multi_aligns.log"


python ${BASE_DIR}/scripts/phylogeny/concatenate_multi_aligns.py \
-o ${PHYLOGENY_OUTPUT_DIR}/alignments/concatenation.fasta
