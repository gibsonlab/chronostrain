#!/bin/bash
set -e
source settings.sh


cd ${BASE_DIR}/scripts
n_reads=10000


analyze_msweep()
{
  mut=$1
  rep=$2
  reads=$3
  trial=$4
  name=$5
  q_theta=$6

  bash msweep/run_msweep.sh $mut $rep $reads $trial 0 "${name}" ${q_theta}
  bash msweep/run_msweep.sh $mut $rep $reads $trial 1 "${name}" ${q_theta}
  bash msweep/run_msweep.sh $mut $rep $reads $trial 2 "${name}" ${q_theta}
  bash msweep/run_msweep.sh $mut $rep $reads $trial 3 "${name}" ${q_theta}
  bash msweep/run_msweep.sh $mut $rep $reads $trial 4 "${name}" ${q_theta}
  bash msweep/run_msweep.sh $mut $rep $reads $trial 5 "${name}" ${q_theta}
}


analyze_mgems()
{
  mut=$1
  rep=$2
  reads=$3
  trial=$4
  name=$5
  q_theta=$6

  bash mgems/analyze_sample_species.sh $mut $rep $reads $trial 0 "${name}" ${q_theta}
  bash mgems/analyze_sample_species.sh $mut $rep $reads $trial 1 "${name}" ${q_theta}
  bash mgems/analyze_sample_species.sh $mut $rep $reads $trial 2 "${name}" ${q_theta}
  bash mgems/analyze_sample_species.sh $mut $rep $reads $trial 3 "${name}" ${q_theta}
  bash mgems/analyze_sample_species.sh $mut $rep $reads $trial 4 "${name}" ${q_theta}
  bash mgems/analyze_sample_species.sh $mut $rep $reads $trial 5 "${name}" ${q_theta}

  bash mgems/analyze_sample_strain.sh $mut $rep $reads $trial 0 "${name}" ${q_theta}
  bash mgems/analyze_sample_strain.sh $mut $rep $reads $trial 1 "${name}" ${q_theta}
  bash mgems/analyze_sample_strain.sh $mut $rep $reads $trial 2 "${name}" ${q_theta}
  bash mgems/analyze_sample_strain.sh $mut $rep $reads $trial 3 "${name}" ${q_theta}
  bash mgems/analyze_sample_strain.sh $mut $rep $reads $trial 4 "${name}" ${q_theta}
  bash mgems/analyze_sample_strain.sh $mut $rep $reads $trial 5 "${name}" ${q_theta}
}


analyze_straingst()
{
  mut=$1
  rep=$2
  reads=$3
  trial=$4
  name=$5
  iters=$6
  min_score=$7

  bash parameter_sensitivity/straingst_run.sh $mut $rep $reads $trial 0 "${name}" ${iters} ${min_score}
  bash parameter_sensitivity/straingst_run.sh $mut $rep $reads $trial 1 "${name}" ${iters} ${min_score}
  bash parameter_sensitivity/straingst_run.sh $mut $rep $reads $trial 2 "${name}" ${iters} ${min_score}
  bash parameter_sensitivity/straingst_run.sh $mut $rep $reads $trial 3 "${name}" ${iters} ${min_score}
  bash parameter_sensitivity/straingst_run.sh $mut $rep $reads $trial 4 "${name}" ${iters} ${min_score}
  bash parameter_sensitivity/straingst_run.sh $mut $rep $reads $trial 5 "${name}" ${iters} ${min_score}
}


for (( trial = 1; trial < ${N_TRIALS}+1; trial++ )); do
  for (( replicate = 1; replicate < ${N_GENOME_REPLICATES}+1; replicate++ )); do
    for mutation_ratio in "${MUTATION_RATIOS[@]}"; do
      # note: p=0.001 is the default setting (assuming it already ran separately), so it is excluded here.
      bash chronostrain/run_chronostrain.sh "$mutation_ratio" "$replicate" "$n_reads" "$trial" "chronostrain_prior1" "0.5"
      bash chronostrain/run_chronostrain.sh "$mutation_ratio" "$replicate" "$n_reads" "$trial" "chronostrain_prior2" "0.1"
      bash chronostrain/run_chronostrain.sh "$mutation_ratio" "$replicate" "$n_reads" "$trial" "chronostrain_prior3" "0.01"
      bash chronostrain/run_chronostrain.sh "$mutation_ratio" "$replicate" "$n_reads" "$trial" "chronostrain_prior4" "0.0001"

      # note: q=0.65 is the default setting for mSWEEP, so it is exluded here.
      analyze_msweep $mutation_ratio $replicate $n_reads $trial "msweep_prior1" 0.5
      analyze_msweep $mutation_ratio $replicate $n_reads $trial "msweep_prior2" 0.35
      analyze_msweep $mutation_ratio $replicate $n_reads $trial "msweep_prior3" 0.8

      # === same for "mgems" pipeline
      analyze_mgems $mutation_ratio $replicate $n_reads $trial "mgems_prior1" 0.5
      analyze_mgems $mutation_ratio $replicate $n_reads $trial "mgems_prior2" 0.35
      analyze_mgems $mutation_ratio $replicate $n_reads $trial "mgems_prior3" 0.8

      # note: n=10, s=0.0 is default for strainGST, so it is excluded here.
      analyze_straingst ${mutation_ratio} ${replicate} ${n_reads} ${trial} "straingst_n20" 20 0.0
      analyze_straingst ${mutation_ratio} ${replicate} ${n_reads} ${trial} "straingst_n10_s01" 10 0.01
      analyze_straingst ${mutation_ratio} ${replicate} ${n_reads} ${trial} "straingst_n10_s02" 10 0.02
      analyze_straingst ${mutation_ratio} ${replicate} ${n_reads} ${trial} "straingst_n10_s03" 10 0.03
      bash parameter_sensitivity/strainge_clean_kmers.sh $mutation_ratio $replicate $n_reads $trial
    done
  done
done