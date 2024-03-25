#!/bin/bash
set -e
source settings.sh
source msweep/settings.sh


echo "[*] Preparing pseudoalignment index"

## ======== next, construct pseudoalignment index.
## Temp dirs
mkdir -p "${THEMISTO_DB_DIR}/_tmp"

## Generate input file for themisto pseudoalignment index construction
echo "[**] Generating input file."

python msweep/themisto_build_input.py \
  -i "${POPPUNK_DB_DIR}/input.tsv" \
  -c "${POPPUNK_DB_DIR}/refine/refine_clusters.csv" \
  -o "${THEMISTO_DB_DIR}"

# invoke kmer index build for pseudoaligner
echo "[**] Building kmer index."
cd "${THEMISTO_DB_DIR}"
themisto build \
  -k 31 \
  -i sequences.txt \
  --index-prefix "${THEMISTO_DB_NAME}" \
  --temp-dir "${THEMISTO_DB_DIR}/_tmp" \
  --mem-gigas 20 --n-threads 4
cd -
