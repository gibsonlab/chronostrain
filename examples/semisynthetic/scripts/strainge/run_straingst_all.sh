#!/bin/bash
set -e
source settings.sh


for (( trial = 1; trial < ${N_TRIALS}+1; trial++ )); do
  for (( replicate = 1; replicate < ${N_GENOME_REPLICATES}+1; replicate++ )); do
    for n_reads in "${SYNTHETIC_COVERAGES[@]}"; do
      for mutation_ratio in "${MUTATION_RATIOS[@]}"; do
        bash strainge/run_straingst.sh $replicate $n_reads $trial 0
        bash strainge/run_straingst.sh $replicate $n_reads $trial 1
        bash strainge/run_straingst.sh $replicate $n_reads $trial 2
        bash strainge/run_straingst.sh $replicate $n_reads $trial 3
        bash strainge/run_straingst.sh $replicate $n_reads $trial 4
      done
    done
	done
done
