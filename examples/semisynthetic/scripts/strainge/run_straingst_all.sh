#!/bin/bash
set -e
source settings.sh


for (( trial = 1; trial < ${N_TRIALS}+1; trial++ )); do
  for (( replicate = 1; replicate < ${N_GENOME_REPLICATES}+1; replicate++ )); do
    for n_reads in "${SYNTHETIC_COVERAGES[@]}"; do
      for mutation_ratio in "${MUTATION_RATIOS[@]}"; do
        bash strainge/run_straingst.sh $mutation_ratio $replicate $n_reads $trial 0 "straingst" 10 0.0
        bash strainge/run_straingst.sh $mutation_ratio $replicate $n_reads $trial 1 "straingst" 10 0.0
        bash strainge/run_straingst.sh $mutation_ratio $replicate $n_reads $trial 2 "straingst" 10 0.0
        bash strainge/run_straingst.sh $mutation_ratio $replicate $n_reads $trial 3 "straingst" 10 0.0
        bash strainge/run_straingst.sh $mutation_ratio $replicate $n_reads $trial 4 "straingst" 10 0.0
        bash strainge/run_straingst.sh $mutation_ratio $replicate $n_reads $trial 5 "straingst" 10 0.0
      done
    done
	done
done
