#!/bin/bash
set -e
source settings.sh
source mutratios/settings.sh


cd ${BASE_DIR}/scripts

for (( trial = 1; trial < ${N_TRIALS}+1; trial++ )); do
  for (( replicate = 1; replicate < ${N_GENOME_REPLICATES}+1; replicate++ )); do
    for n_reads in "${SYNTHETIC_COVERAGES[@]}"; do
      for mutation_ratio in "${MUTATION_RATIOS[@]}"; do
        if [ "${mutation_ratio}" == "1.0" ]; then continue; fi  # assume these analyses are already done separately.

        bash mutratios/filter.sh "$mutation_ratio" "$replicate" "$n_reads" "$trial"  # custom script.
        bash chronostrain/run_chronostrain.sh "$mutation_ratio" "$replicate" "$n_reads" "$trial" "chronostrain" "0.001"

        bash mutratios/msweep_pseudoalignment.sh $mutation_ratio $replicate $n_reads $trial
        bash mutratios/msweep_run.sh $mutation_ratio $replicate $n_reads $trial 0
        bash mutratios/msweep_run.sh $mutation_ratio $replicate $n_reads $trial 1
        bash mutratios/msweep_run.sh $mutation_ratio $replicate $n_reads $trial 2
        bash mutratios/msweep_run.sh $mutation_ratio $replicate $n_reads $trial 3
        bash mutratios/msweep_run.sh $mutation_ratio $replicate $n_reads $trial 4
        bash mutratios/msweep_run.sh $mutation_ratio $replicate $n_reads $trial 5

        bash strainge/run_straingst.sh $mutation_ratio $replicate $n_reads $trial 0 "straingst" 10 0.0
        bash strainge/run_straingst.sh $mutation_ratio $replicate $n_reads $trial 1 "straingst" 10 0.0
        bash strainge/run_straingst.sh $mutation_ratio $replicate $n_reads $trial 2 "straingst" 10 0.0
        bash strainge/run_straingst.sh $mutation_ratio $replicate $n_reads $trial 3 "straingst" 10 0.0
        bash strainge/run_straingst.sh $mutation_ratio $replicate $n_reads $trial 4 "straingst" 10 0.0
        bash strainge/run_straingst.sh $mutation_ratio $replicate $n_reads $trial 5 "straingst" 10 0.0

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