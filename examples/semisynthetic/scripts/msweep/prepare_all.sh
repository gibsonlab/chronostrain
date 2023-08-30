#!/bin/bash
set -e
source settings.sh

cd ${BASE_DIR}/scripts
for mutation_ratio in "${MUTATION_RATIOS[@]}"; do
  for (( replicate = 1; replicate < ${N_GENOME_REPLICATES}+1; replicate++ )); do
    bash msweep/prepare_pseudoalignment_index.sh "${mutation_ratio}" "${replicate}"
  done
done
