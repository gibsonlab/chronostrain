#!/bin/bash
set -e

source settings.sh

mkdir -p $READS_DIR
SEED=123

export CHRONOSTRAIN_LOG_FILEPATH="${CHRONOSTRAIN_DATA_DIR}/logs/reads_${N_READS}/qs_${QUALITY_SHIFT}/single-run/readgen.log"

python ${PROJECT_DIR}/scripts/readgen.py \
--num_reads $N_READS \
--read_len $READ_LEN \
--out_dir $READS_DIR \
--profiles $READ_PROFILE_PATH $READ_PROFILE_PATH \
--abundance_path $TRUE_ABUNDANCE_PATH \
--seed $SEED \
--qShift ${QUALITY_SHIFT} \
--num_cores 1
