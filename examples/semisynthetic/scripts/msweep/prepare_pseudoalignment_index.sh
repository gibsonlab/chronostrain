#!/bin/bash
set -e
source settings.sh


## ======== next, construct pseudoalignment index.
## Temp dirs
mkdir -p ${THEMISTO_DB_DIR}/_tmp

## Generate input file for themisto pseudoalignment index construction
echo "[*] Generating input file."

python msweep/themisto_build_input.py \
  -i ${REFSEQ_INDEX} \
  -g /mnt/e/semisynthetic_data/sim_genomes/NZ_CP024859.1.sim_mutant.fasta \
  -g /mnt/e/semisynthetic_data/sim_genomes/NZ_CP092452.1.sim_mutant.fasta \
  -c ${THEMISTO_DB_DIR}/poppunk/refine/refine_clusters.csv \
  -o ${THEMISTO_DB_DIR}

# invoke kmer index build for pseudoaligner
echo "[*] Building kmer index."
cd ${THEMISTO_DB_DIR}
${THEMISTO_BIN_DIR}/themisto build \
  -k 31 \
  -i sequences.txt \
  --index-prefix enterobacteriaceae \
  --temp-dir ${THEMISTO_DB_DIR}/_tmp \
  --mem-gigas 20 --n-threads 8
cd -
