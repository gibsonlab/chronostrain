#!/bin/bash
set -e
source settings.sh
source strainge/settings.sh

# ============ Requires arguments:
mutation_ratio=$1
replicate=$2
n_reads=$3
trial=$4

require_variable "mutation_ratio" $mutation_ratio
require_variable "replicate" $replicate
require_variable "n_reads" $n_reads
require_variable "trial" $trial


trial_dir=$(get_trial_dir "${mutation_ratio}" "$replicate" "$n_reads" "$trial")
read_dir=${trial_dir}/reads
kmer_dir=${trial_dir}/output/straingst_kmers


# ========== Run
echo "[*] Cleaning up kmer directory: ${kmer_dir}"
rm -rf ${kmer_dir}