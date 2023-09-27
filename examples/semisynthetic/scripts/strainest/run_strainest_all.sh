#!/bin/bash
set -e
source settings.sh


cd ${BASE_DIR}/scripts

for (( trial = 1; trial < ${N_TRIALS}+1; trial++ )); do
  for (( replicate = 1; replicate < ${N_GENOME_REPLICATES}+1; replicate++ )); do
    for n_reads in "${SYNTHETIC_COVERAGES[@]}"; do
      for mutation_ratio in "${MUTATION_RATIOS[@]}"; do
        bash strainest/run_strainest.sh $mutation_ratio $replicate $n_reads $trial 0
        bash strainest/run_strainest.sh $mutation_ratio $replicate $n_reads $trial 1
        bash strainest/run_strainest.sh $mutation_ratio $replicate $n_reads $trial 2
        bash strainest/run_strainest.sh $mutation_ratio $replicate $n_reads $trial 3
        bash strainest/run_strainest.sh $mutation_ratio $replicate $n_reads $trial 4
        bash strainest/run_strainest.sh $mutation_ratio $replicate $n_reads $trial 5
      done
    done
	done
done
