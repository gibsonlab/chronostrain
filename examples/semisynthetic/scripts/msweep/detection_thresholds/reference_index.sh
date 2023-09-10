#!/bin/bash
set -e
source settings.sh
source msweep/settings.sh


echo "[*] Preparing pseudoalignment index for references only."

## ======== next, construct pseudoalignment index.
## Temp dirs
themisto_db_dir="/mnt/e/semisynthetic_data/msweep_thresholds/pseudoalignment_index"
mkdir -p "${themisto_db_dir}/_tmp"

## Generate input file for themisto pseudoalignment index construction
echo "[**] Generating input file."

python msweep/themisto_build_input.py \
  -i "${REFSEQ_INDEX}" \
  -c "${POPPUNK_REFSEQ_DIR}/refine/refine_clusters.csv" \
  -o "${themisto_db_dir}"

# invoke kmer index build for pseudoaligner
echo "[**] Building kmer index."
cd "${themisto_db_dir}"
${THEMISTO_BIN_DIR}/themisto build \
  -k 31 \
  -i sequences.txt \
  --index-prefix enterobacteriaceae \
  --temp-dir "${themisto_db_dir}/_tmp" \
  --mem-gigas 20 --n-threads 8
cd -