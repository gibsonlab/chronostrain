#!/bin/bash
set -e
source settings.sh


mutation_ratio=$1
replicate=$2
require_variable "mutation_ratio" $mutation_ratio
require_variable "replicate" $replicate


replicate_dir=$(get_replicate_dir "${mutation_ratio}" "${replicate}")
python dataset/append_chronostrain_json.py \
  -i ${CHRONOSTRAIN_DB_JSON_SRC} \
  -o ${replicate_dir}/databases/chronostrain/ecoli.json \
  -s NZ_CP051001.1 \
  -m NZ_CP051001.1.sim_mutant \
  -s NZ_CP068279.1 \
  -m NZ_CP068279.1.sim_mutant \
  -s NZ_CP035882.1 \
  -m NZ_CP035882.1.sim_mutant \
  -g ${replicate_dir}/sim_genomes \
  -ds ${CHRONOSTRAIN_DB_DIR_SRC} \
  -dt ${replicate_dir}/databases/chronostrain

