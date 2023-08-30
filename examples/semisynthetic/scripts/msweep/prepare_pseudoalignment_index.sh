#!/bin/bash
set -e
source settings.sh
source msweep/settings.sh


mutation_ratio=$1
replicate=$2
require_variable "mutation_ratio" $mutation_ratio
require_variable "replicate" $replicate


echo "[*] Preparing pseudoalignment index for mut_ratio ${mutation_ratio}, replicate ${replicate}"

## ======== next, construct pseudoalignment index.
## Temp dirs
replicate_dir=$(get_replicate_dir "${mutation_ratio}" "${replicate}")
themisto_db_dir=$(get_themisto_db_dir "${mutation_ratio}" "${replicate}")
mkdir -p "${themisto_db_dir}/_tmp"

## Generate input file for themisto pseudoalignment index construction
echo "[**] Generating input file."

python msweep/themisto_build_input.py \
  -i ${REFSEQ_INDEX} \
  -g ${replicate_dir}/sim_genomes/NZ_CP022154.1.sim_mutant.fasta \
  -g ${replicate_dir}/sim_genomes/NZ_LR536430.1.sim_mutant.fasta \
  -c "${POPPUNK_REFSEQ_DIR}/refine/refine_clusters.csv" \
  -o "${themisto_db_dir}"

# invoke kmer index build for pseudoaligner
echo "[**] Building kmer index."
cd "${themisto_db_dir}"
${THEMISTO_BIN_DIR}/themisto build \
  -k 31 \
  -i sequences.txt \
  --index-prefix enterobacteriaceae \
  --temp-dir "${themisto_db_dir}/_tmp" \
  --mem-gigas 20 --n-threads 8
cd -
