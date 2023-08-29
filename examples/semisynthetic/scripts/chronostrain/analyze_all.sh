#!/bin/bash
set -e
source settings.sh


cd ${BASE_DIR}/scripts

for (( replicate = 1; replicate < ${N_GENOME_REPLICATES}+1; replicate++ )); do
  for (( trial = 1; trial < ${N_TRIALS}+1; trial++ )); do
    for n_reads in "${SYNTHETIC_COVERAGES[@]}"; do
      bash chronostrain/filter.sh $replicate $n_reads $trial
      bash chronostrain/run_chronostrain.sh $replicate $n_reads $trial
    done
  done
done
