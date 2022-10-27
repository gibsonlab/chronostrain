#!/bin/bash
set -e
source settings.sh

# ======================================== Functions ===============================
check_program 'git'
check_program 'python'

if [ ! -d ncbi-genome-download ]; then
	echo "[*] Cloning repository 'ncbi-genome-download' (from author 'kblin')."
	git clone https://github.com/kblin/ncbi-genome-download.git
fi

echo "[*] Using tool to download complete assemblies of Klebsiella from NCBI (# cores = ${NUM_CORES})."
mkdir -p ${NCBI_REFSEQ_DIR}
cd ncbi-genome-download
python ncbi-genome-download-runner.py bacteria \
	-l complete -g Klebsiella -H -F all \
	-o ${NCBI_REFSEQ_DIR} -v --parallel ${NUM_CORES} --progress-bar

echo "[*] Indexing reference sequences."
python index_refseq.py -r ${NCBI_REFSEQ_DIR} -o ${INDEX_FILE}
echo "[*] Finished indexing. Wrote index to: ${INDEX_FILE}"
