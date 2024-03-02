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

      # note: q=0.65 is the default setting for mSWEEP, so it is exluded here.
      bash msweep/run_msweep.sh $mutation_ratio $replicate $n_reads $trial 0 "msweep_prior1" 0.5
      bash msweep/run_msweep.sh $mutation_ratio $replicate $n_reads $trial 1 "msweep_prior1" 0.5
      bash msweep/run_msweep.sh $mutation_ratio $replicate $n_reads $trial 2 "msweep_prior1" 0.5
      bash msweep/run_msweep.sh $mutation_ratio $replicate $n_reads $trial 3 "msweep_prior1" 0.5
      bash msweep/run_msweep.sh $mutation_ratio $replicate $n_reads $trial 4 "msweep_prior1" 0.5
      bash msweep/run_msweep.sh $mutation_ratio $replicate $n_reads $trial 5 "msweep_prior1" 0.5

      bash msweep/run_msweep.sh $mutation_ratio $replicate $n_reads $trial 0 "msweep_prior2" 0.35
      bash msweep/run_msweep.sh $mutation_ratio $replicate $n_reads $trial 1 "msweep_prior2" 0.35
      bash msweep/run_msweep.sh $mutation_ratio $replicate $n_reads $trial 2 "msweep_prior2" 0.35
      bash msweep/run_msweep.sh $mutation_ratio $replicate $n_reads $trial 3 "msweep_prior2" 0.35
      bash msweep/run_msweep.sh $mutation_ratio $replicate $n_reads $trial 4 "msweep_prior2" 0.35
      bash msweep/run_msweep.sh $mutation_ratio $replicate $n_reads $trial 5 "msweep_prior2" 0.35

      bash msweep/run_msweep.sh $mutation_ratio $replicate $n_reads $trial 0 "msweep_prior3" 0.8
      bash msweep/run_msweep.sh $mutation_ratio $replicate $n_reads $trial 1 "msweep_prior3" 0.8
      bash msweep/run_msweep.sh $mutation_ratio $replicate $n_reads $trial 2 "msweep_prior3" 0.8
      bash msweep/run_msweep.sh $mutation_ratio $replicate $n_reads $trial 3 "msweep_prior3" 0.8
      bash msweep/run_msweep.sh $mutation_ratio $replicate $n_reads $trial 4 "msweep_prior3" 0.8
      bash msweep/run_msweep.sh $mutation_ratio $replicate $n_reads $trial 5 "msweep_prior3" 0.8

      # note: n=10, s=0.0 is default for strainGST, so it is excluded here.
      bash parameter_sensitivity/straingst_run.sh $mutation_ratio $replicate $n_reads $trial 0 "straingst_n20" 20 0.0
      bash parameter_sensitivity/straingst_run.sh $mutation_ratio $replicate $n_reads $trial 1 "straingst_n20" 20 0.0
      bash parameter_sensitivity/straingst_run.sh $mutation_ratio $replicate $n_reads $trial 2 "straingst_n20" 20 0.0
      bash parameter_sensitivity/straingst_run.sh $mutation_ratio $replicate $n_reads $trial 3 "straingst_n20" 20 0.0
      bash parameter_sensitivity/straingst_run.sh $mutation_ratio $replicate $n_reads $trial 4 "straingst_n20" 20 0.0
      bash parameter_sensitivity/straingst_run.sh $mutation_ratio $replicate $n_reads $trial 5 "straingst_n20" 20 0.0

      bash parameter_sensitivity/straingst_run.sh $mutation_ratio $replicate $n_reads $trial 0 "straingst_n10_s01" 10 0.01
      bash parameter_sensitivity/straingst_run.sh $mutation_ratio $replicate $n_reads $trial 1 "straingst_n10_s01" 10 0.01
      bash parameter_sensitivity/straingst_run.sh $mutation_ratio $replicate $n_reads $trial 2 "straingst_n10_s01" 10 0.01
      bash parameter_sensitivity/straingst_run.sh $mutation_ratio $replicate $n_reads $trial 3 "straingst_n10_s01" 10 0.01
      bash parameter_sensitivity/straingst_run.sh $mutation_ratio $replicate $n_reads $trial 4 "straingst_n10_s01" 10 0.01
      bash parameter_sensitivity/straingst_run.sh $mutation_ratio $replicate $n_reads $trial 5 "straingst_n10_s01" 10 0.01

      bash parameter_sensitivity/straingst_run.sh $mutation_ratio $replicate $n_reads $trial 0 "straingst_n10_s02" 10 0.02
      bash parameter_sensitivity/straingst_run.sh $mutation_ratio $replicate $n_reads $trial 1 "straingst_n10_s02" 10 0.02
      bash parameter_sensitivity/straingst_run.sh $mutation_ratio $replicate $n_reads $trial 2 "straingst_n10_s02" 10 0.02
      bash parameter_sensitivity/straingst_run.sh $mutation_ratio $replicate $n_reads $trial 3 "straingst_n10_s02" 10 0.02
      bash parameter_sensitivity/straingst_run.sh $mutation_ratio $replicate $n_reads $trial 4 "straingst_n10_s02" 10 0.02
      bash parameter_sensitivity/straingst_run.sh $mutation_ratio $replicate $n_reads $trial 5 "straingst_n10_s02" 10 0.02

      bash parameter_sensitivity/straingst_run.sh $mutation_ratio $replicate $n_reads $trial 0 "straingst_n10_s03" 10 0.03
      bash parameter_sensitivity/straingst_run.sh $mutation_ratio $replicate $n_reads $trial 1 "straingst_n10_s03" 10 0.03
      bash parameter_sensitivity/straingst_run.sh $mutation_ratio $replicate $n_reads $trial 2 "straingst_n10_s03" 10 0.03
      bash parameter_sensitivity/straingst_run.sh $mutation_ratio $replicate $n_reads $trial 3 "straingst_n10_s03" 10 0.03
      bash parameter_sensitivity/straingst_run.sh $mutation_ratio $replicate $n_reads $trial 4 "straingst_n10_s03" 10 0.03
      bash parameter_sensitivity/straingst_run.sh $mutation_ratio $replicate $n_reads $trial 5 "straingst_n10_s03" 10 0.03

      bash parameter_sensitivity/strainge_clean_kmers.sh $mutation_ratio $replicate $n_reads $trial
    done
  done
done