#!/bin/bash
set -e
source ../scripts/settings.sh
export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/concatenate_multi_aligns.log"


python concatenate_multi_aligns.py \
-o ${CHRONOSTRAIN_DB_DIR}/phylogeny/concatenation.fasta
