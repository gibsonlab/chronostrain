#!/bin/bash
set -e
source settings.sh
source mutratios/settings.sh

seed=0
require_program 'art_illumina'
require_program 'prefetch'
require_program 'fasterq-dump'


# This script skips the background reads' fetching process. (They are added after filtering)
# =============== Sample synthetic reads
for (( trial = 1; trial < ${N_TRIALS}+1; trial++ )); do
  for (( replicate = 1; replicate < ${N_GENOME_REPLICATES}+1; replicate++ )); do
    for n_reads in "${SYNTHETIC_COVERAGES[@]}"; do
      for mutation_ratio in "${MUTATION_RATIOS[@]}"; do
        replicate_dir=$(get_replicate_dir "${mutation_ratio}" "${replicate}")
        replicate_ground_truth=${replicate_dir}/genomes/abundances.txt
        seed=$((seed+1))

        trial_dir=$(get_trial_dir $mutation_ratio $replicate $n_reads $trial)
        read_dir=${trial_dir}/reads
        breadcrumb=${trial_dir}/read_sample.DONE

        if [[ -f "${breadcrumb}" ]]; then
          echo "[*] Skipping reads: ${n_reads}, trial #${trial}] -> ${trial_dir}"
        else
          echo "[*] Sampling [Number of reads: ${n_reads}, trial #${trial}] -> ${trial_dir}"

          mkdir -p $read_dir
          export CHRONOSTRAIN_DB_JSON=${replicate_dir}/databases/chronostrain/ecoli.json
          export CHRONOSTRAIN_DB_DIR=${replicate_dir}/databases/chronostrain
          export CHRONOSTRAIN_LOG_FILEPATH="${read_dir}/read_simulation.log"
          export CHRONOSTRAIN_CACHE_DIR="${trial_dir}/cache"

          python ${BASE_DIR}/helpers/sample_reads.py \
          --out_dir $read_dir \
          --abundance_path $replicate_ground_truth \
          --genome_dir ${replicate_dir}/genomes \
          --num_reads $n_reads \
          --read_len $READ_LEN \
          --seed ${seed} \
          --num_cores $N_CORES

          touch "${breadcrumb}"
        fi
      done
    done
  done
done