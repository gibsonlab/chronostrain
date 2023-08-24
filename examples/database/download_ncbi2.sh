#!/bin/bash
set -e
source settings.sh

# =============== Similar to download_ncbi, but uses NCBI's native API tool.

require_program 'python'
require_program 'datasets'  # NCBI datasets: https://www.ncbi.nlm.nih.gov/datasets/docs/v2/reference-docs/command-line/datasets

require_variable 'TARGET_TAXA' $TARGET_TAXA
require_variable 'NCBI_REFSEQ_DIR' $NCBI_REFSEQ_DIR
require_variable 'REFSEQ_INDEX' $REFSEQ_INDEX


mkdir -p ${NCBI_REFSEQ_DIR}
env \
  CHRONOSTRAIN_LOG_FILEPATH="${run_dir}/inference.log" \
  CHRONOSTRAIN_DB_DIR=. \
  CHRONOSTRAIN_INI=./chronostrain.ini \
  python download_ncbi.py -t "${TARGET_TAXA}" -d ${NCBI_REFSEQ_DIR} -o ${REFSEQ_INDEX}
