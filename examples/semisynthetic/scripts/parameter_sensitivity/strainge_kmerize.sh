#!/bin/bash
set -e
source settings.sh
source strainge/settings.sh

# ============ Requires arguments:
mutation_ratio=$1
replicate=$2
n_reads=$3
trial=$4
time_point=$5

require_variable "mutation_ratio" $mutation_ratio
require_variable "replicate" $replicate
require_variable "n_reads" $n_reads
require_variable "trial" $trial
require_variable "time_point" $time_point


trial_dir=$(get_trial_dir "${mutation_ratio}" "$replicate" "$n_reads" "$trial")
read_dir=${trial_dir}/reads
kmer_dir=${trial_dir}/output/straingst_kmers


# ========== Run
echo "[*] Running k-merization for (replicate: ${replicate} | n_reads: ${n_reads} | trial: ${trial} | timepoint #${time_point})"
read_kmers=${kmer_dir}/reads.${time_point}.hdf5
mkdir -p ${kmer_dir}


echo "[*] Kmerizing..."
straingst kmerize \
-k 23 \
-o ${read_kmers} \
${read_dir}/${time_point}_sim_1.fq \
${BACKGROUND_FASTQ_DIR}/${time_point}_background_1.fq \
${read_dir}/${time_point}_sim_2.fq \
${BACKGROUND_FASTQ_DIR}/${time_point}_background_2.fq
