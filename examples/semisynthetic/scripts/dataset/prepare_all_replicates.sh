#!/bin/bash
set -e
source settings.sh


cd ${BASE_DIR}/scripts

seed=101

for (( replicate = 1; replicate < ${N_GENOME_REPLICATES}+1; replicate++ )); do
  echo "[*] Preparing genomes for replicate ${replicate}."
  bash dataset/prepare_genomes.sh "${seed}${replicate}" $replicate
done
