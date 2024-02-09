#!/bin/bash
set -e
source settings.sh


cd ${BASE_DIR}/scripts
n_reads=10000


for (( trial = 1; trial < ${N_TRIALS}+1; trial++ )); do
  for (( replicate = 1; replicate < ${N_GENOME_REPLICATES}+1; replicate++ )); do
    for mutation_ratio in "${MUTATION_RATIOS[@]}"; do
      # note: p=0.001 is the default setting (assuming it already ran separately), so it is excluded here.
      bash chronostrain/run_chronostrain.sh "$mutation_ratio" "$replicate" "$n_reads" "$trial" "chronostrain_prior1" "0.5"
      bash chronostrain/run_chronostrain.sh "$mutation_ratio" "$replicate" "$n_reads" "$trial" "chronostrain_prior2" "0.1"
      bash chronostrain/run_chronostrain.sh "$mutation_ratio" "$replicate" "$n_reads" "$trial" "chronostrain_prior3" "0.01"
      bash chronostrain/run_chronostrain.sh "$mutation_ratio" "$replicate" "$n_reads" "$trial" "chronostrain_prior4" "0.0001"
    done
  done
done