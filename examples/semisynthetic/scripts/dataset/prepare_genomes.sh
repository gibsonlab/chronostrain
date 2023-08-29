#!/bin/bash
set -e
source settings.sh


base_seed=$1
replicate=$2
require_variable "base_seed" $base_seed
require_variable "replicate" $replicate


replicate_dir=${DATA_DIR}/replicate_${replicate}
genome_dir=${replicate_dir}/sim_genomes
mkdir -p $genome_dir

# Copy existing reference genomes
while IFS=$'\t' read -r -a columns
do
  acc="${columns[3]}"
  seq_path="${columns[5]}"
  if [ "${acc}" == "NZ_CP022154.1" ] || [ "${acc}" == "NZ_LR536430.1" ]; then
    ln -s ${seq_path} ${genome_dir}/${acc}.fasta
  fi
done < ${REFSEQ_INDEX}


create_mutant()
{
  src_acc=$1
  tgt_acc=$2
  seed=$3
  mutation_rate=$4
  db_json=$5
  python ${BASE_DIR}/helpers/mutate_genome.py \
    -i ${genome_dir}/${src_acc}.fasta \
    -o ${genome_dir}/${tgt_acc}.fasta \
    -j ${db_json} \
    -d ${mutation_rate} \
    -sid ${src_acc} \
    -tid ${tgt_acc} \
    -s ${seed}
}


# ======================== genomes to include in database
create_mutant NZ_CP022154.1 NZ_CP022154.1.sim_mutant "${base_seed}1" 0.002 ${CHRONOSTRAIN_DB_JSON_SRC}
create_mutant NZ_LR536430.1 NZ_LR536430.1.sim_mutant "${base_seed}2" 0.002 ${CHRONOSTRAIN_DB_JSON_SRC}


# ======================== Update chronostrain JSON.
bash dataset/append_chronostrain_json.sh $replicate  # this initializes the JSON file at CHRONOSTRAIN_DB_JSON


# ======================== genomes to simulate reads from
db_json_replicate=${replicate_dir}/databases/chronostrain/ecoli.json
create_mutant NZ_CP022154.1 NZ_CP022154.1.READSIM_MUTANT "${base_seed}3" 0.001 ${db_json_replicate}
create_mutant NZ_LR536430.1 NZ_LR536430.1.READSIM_MUTANT "${base_seed}4" 0.001 ${db_json_replicate}
create_mutant NZ_CP022154.1.sim_mutant NZ_CP022154.1.sim_mutant.READSIM_MUTANT "${base_seed}5" 0.001 ${db_json_replicate}
create_mutant NZ_LR536430.1.sim_mutant NZ_LR536430.1.sim_mutant.READSIM_MUTANT "${base_seed}6" 0.001 ${db_json_replicate}
