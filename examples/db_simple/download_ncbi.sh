#!/bin/bash
set -e
source settings.sh

# ======================================== Functions ===============================
# REQUIRES: sratools (prefetch + fasterq-dump)
check_program 'ncbi-genome-download'
check_program 'fasterq-dump'


mkdir -p ${NCBI_REFSEQ_DIR}
cd ~/ncbi-genome-download  # TODO replace this with forked github branch.
python ncbi-genome-download-runner.py bacteria -l complete -g Klebsiella -H -F all -o ${NCBI_REFSEQ_DIR} -v --parallel 1 --progress-bar