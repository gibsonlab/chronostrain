#!/bin/bash
set -e
source settings.sh

cd ${BASE_DIR}/scripts

for (( trial = 1; trial < ${N_TRIALS}+1; trial++ )); do
  for (( replicate = 1; replicate < ${N_GENOME_REPLICATES}+1; replicate++ )); do
    for n_reads in "${SYNTHETIC_COVERAGES[@]}"; do
      for mutation_ratio in "${MUTATION_RATIOS[@]}"; do
        echo "[*] Performing mGEMS analysis pipeline (mut_ratio: ${mutation_ratio} | replicate: ${replicate} |  n_reads: ${n_reads} | trial: ${trial})"
        bash mgems/analyze_sample_species.sh $mutation_ratio $replicate $n_reads $trial 0 "mgems" 0.65
        bash mgems/analyze_sample_species.sh $mutation_ratio $replicate $n_reads $trial 1 "mgems" 0.65
        bash mgems/analyze_sample_species.sh $mutation_ratio $replicate $n_reads $trial 2 "mgems" 0.65
        bash mgems/analyze_sample_species.sh $mutation_ratio $replicate $n_reads $trial 3 "mgems" 0.65
        bash mgems/analyze_sample_species.sh $mutation_ratio $replicate $n_reads $trial 4 "mgems" 0.65
        bash mgems/analyze_sample_species.sh $mutation_ratio $replicate $n_reads $trial 5 "mgems" 0.65

        bash mgems/analyze_sample_strain.sh $mutation_ratio $replicate $n_reads $trial 0 "mgems" 0.65
        bash mgems/analyze_sample_strain.sh $mutation_ratio $replicate $n_reads $trial 1 "mgems" 0.65
        bash mgems/analyze_sample_strain.sh $mutation_ratio $replicate $n_reads $trial 2 "mgems" 0.65
        bash mgems/analyze_sample_strain.sh $mutation_ratio $replicate $n_reads $trial 3 "mgems" 0.65
        bash mgems/analyze_sample_strain.sh $mutation_ratio $replicate $n_reads $trial 4 "mgems" 0.65
        bash mgems/analyze_sample_strain.sh $mutation_ratio $replicate $n_reads $trial 5 "mgems" 0.65
      done
    done
  done
done
