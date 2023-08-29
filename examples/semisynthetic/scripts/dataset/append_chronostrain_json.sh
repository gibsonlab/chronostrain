#!/bin/bash
set -e
source settings.sh


replicate=$1
require_variable "replicate" $replicate


replicate_dir=${DATA_DIR}/replicate_${replicate}
python dataset/append_chronostrain_json.py \
  -i ${CHRONOSTRAIN_DB_JSON_SRC} \
  -o ${replicate_dir}/databases/chronostrain/ecoli.json \
  -s NZ_CP022154.1 \
  -m NZ_CP022154.1.sim_mutant \
  -s NZ_LR536430.1 \
  -m NZ_LR536430.1.sim_mutant \
  -g ${replicate_dir}/sim_genomes \
  -ds ${CHRONOSTRAIN_DB_DIR_SRC} \
  -dt ${replicate_dir}/databases/chronostrain

