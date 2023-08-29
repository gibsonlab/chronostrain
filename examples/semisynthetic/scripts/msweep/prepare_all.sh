#!/bin/bash
set -e
source settings.sh

cd ${BASE_DIR}/scripts
for (( replicate = 1; replicate < ${N_GENOME_REPLICATES}+1; replicate++ )); do
  bash msweep/prepare_pseudoalignment_index.sh "${replicate}"
done
