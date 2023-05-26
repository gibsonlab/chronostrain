#!/bin/bash
set -e
source settings.sh


ST_OUTPUT=${DATA_DIR}/fastmlst.tsv
fastmlst -s '\t' ${DATA_DIR}/*/isolate_assemblies/*.fasta > ${ST_OUTPUT}
