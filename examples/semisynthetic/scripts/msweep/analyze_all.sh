#!/bin/bash
set -e
source settings.sh

cd ${BASE_DIR}/scripts

for (( replicate = 1; replicate < ${N_GENOME_REPLICATES}+1; replicate++ )); do
  for (( trial = 1; trial < ${N_TRIALS}+1; trial++ )); do
    for n_reads in "${SYNTHETIC_COVERAGES[@]}"; do
      bash msweep/run_pseudoalignment.sh $replicate $n_reads $trial

      bash msweep/run_msweep.sh $replicate $n_reads $trial 0
      bash msweep/run_msweep.sh $replicate $n_reads $trial 1
      bash msweep/run_msweep.sh $replicate $n_reads $trial 2
      bash msweep/run_msweep.sh $replicate $n_reads $trial 3
      bash msweep/run_msweep.sh $replicate $n_reads $trial 4
    done
  done
done
