#!/bin/bash
set -e
source settings.sh

# ======================================== Functions ===============================
# REQUIRES: sratools (prefetch + fasterq-dump)
check_program 'ncbi-genome-download'
check_program 'fasterq-dump'


mkdir -p ${NCBI_REFSEQ_DIR}
ncbi-genome-download bacteria -l complete -g Gemmiger,Sutterella,Ruminococcus -H -F all -o ${NCBI_REFSEQ_DIR}