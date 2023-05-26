#!/bin/bash
set -e
source settings.sh

# ======================================== Functions ===============================
require_program 'python'
require_program 'ncbi-genome-download'

require_variable 'NUM_CORES' $NUM_CORES
require_variable 'NCBI_REFSEQ_DIR' $NCBI_REFSEQ_DIR
require_variable 'TARGET_TAXA' $TARGET_TAXA
require_variable 'REFSEQ_INDEX' $REFSEQ_INDEX

echo "[*] Using tool to download complete assemblies from NCBI (# cores = ${NUM_CORES})."
mkdir -p ${NCBI_REFSEQ_DIR}
ncbi-genome-download bacteria \
	-l complete \
	-g "${TARGET_TAXA}" \
	-H -F all \
	-o ${NCBI_REFSEQ_DIR} \
	-v \
	--parallel ${NUM_CORES} \
	--progress-bar

echo "[*] Indexing reference sequences."
python index_refseq.py -r ${NCBI_REFSEQ_DIR} -o ${REFSEQ_INDEX}
echo "[*] Finished indexing. Wrote index to: ${REFSEQ_INDEX}"
