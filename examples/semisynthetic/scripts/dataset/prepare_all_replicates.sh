#!/bin/bash
set -e
source settings.sh


cd ${BASE_DIR}/scripts

seed=101

m_idx=0
for mutation_ratio in "${MUTATION_RATIOS[@]}"; do
  for (( replicate = 1; replicate < ${N_GENOME_REPLICATES}+1; replicate++ )); do
    echo "[*] Preparing genomes for mut_ratio ${mutation_ratio}, replicate ${replicate}."
    bash dataset/prepare_genomes.sh "${seed}${m_idx}${replicate}" "$mutation_ratio" "$replicate"
  done
  m_idx=$(( ${m_idx}+1 ))
done
