#!/bin/bash
set -e
source settings.sh

# ======================================== Functions ===============================
# REQUIRES: sratools (prefetch + fasterq-dump)
require_program 'ncbi-genome-download'
require_program 'fasterq-dump'


mkdir -p ${NCBI_REFSEQ_DIR}
cd ~/ncbi-genome-download  # TODO replace this with forked github branch.
python ncbi-genome-download-runner.py bacteria -l complete \
-g Biostraticola,Buttiauxella,Cedecea,Citrobacter,Cronobacter,Enterobacillus,Enterobacter,Escherichia,Franconibacter,Gibbsiella,Izhakiella,Klebsiella,Kluyvera,Kosakonia,Leclercia,Lelliottia,Limnobaculum,Mangrovibacter,Metakosakonia,Phytobacter,Pluralibacter,Pseudescherichia,Pseudocitrobacter,Raoultella,Rosenbergiella,Saccharobacter,Salmonella,Scandinavium,Shigella,Shimwellia,Siccibacter,Trabulsiella,Yokenella \
-H -F all -o ${NCBI_REFSEQ_DIR} -v --parallel 1 --progress-bar