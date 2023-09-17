#!/bin/bash
set -e
source settings.sh


base_seed=$1
mutation_ratio=$2
replicate=$3
require_variable "base_seed" $base_seed
require_variable "mutation_ratio" $mutation_ratio
require_variable "replicate" $replicate
require_program bc

replicate_dir=$(get_replicate_dir "${mutation_ratio}" "${replicate}")
genome_dir=${replicate_dir}/genomes
mkdir -p "$genome_dir"

## Copy existing reference genomes
#while IFS=$'\t' read -r -a columns
#do
#  acc="${columns[3]}"
#  seq_path="${columns[5]}"
#  if [ "${acc}" == "NZ_CP051001.1" ] || [ "${acc}" == "NZ_CP068279.1" ] || [ "${acc}" == "NZ_CP035882.1" ]; then
#    ln -s ${seq_path} ${genome_dir}/${acc}.fasta
#  fi
#done < ${REFSEQ_INDEX}


create_mutant()
{
  src_acc=$1
  tgt_acc=$2
  sim_seed=$3
  mutation_rate=$4
  marker_mutation_rate=$5
  db_json=$6
  echo "Mutating ${src_acc}, seed=${sim_seed}"
  python dataset/mutate_genome.py \
    -i ${genome_dir}/${src_acc}.fasta \
    -o ${genome_dir}/${tgt_acc}.fasta \
    -j ${db_json} \
    -d ${mutation_rate} \
    -dm ${marker_mutation_rate} \
    -sid ${src_acc} \
    -tid ${tgt_acc} \
    -s ${sim_seed}
}


# ======================== genomes to include in database
#genome_rate=${BASE_GENOME_MUTATION_RATE}
#marker_rate=$(echo "scale=10; ${genome_rate} * ${mutation_ratio}" | bc)
#echo "[*] genome_rate = ${genome_rate}, marker_rate = ${marker_rate} (mutation ratio = ${mutation_ratio})"
#create_mutant NZ_CP051001.1 NZ_CP051001.1.sim_mutant "${base_seed}1" ${genome_rate} ${marker_rate} ${CHRONOSTRAIN_DB_JSON_SRC}
#create_mutant NZ_CP068279.1 NZ_CP068279.1.sim_mutant "${base_seed}2" ${genome_rate} ${marker_rate} ${CHRONOSTRAIN_DB_JSON_SRC}
#create_mutant NZ_CP035882.1 NZ_CP035882.1.sim_mutant "${base_seed}3" ${genome_rate} ${marker_rate} ${CHRONOSTRAIN_DB_JSON_SRC}


# ======================== Update chronostrain JSON.
#bash dataset/append_chronostrain_json.sh ${mutation_ratio} ${replicate}  # this initializes the JSON file at CHRONOSTRAIN_DB_JSON


# ======================== Pick random genomes to simulate reads from.
python dataset/pick_random_genomes.py \
  -i ${REFSEQ_INDEX} \
  -p /mnt/e/semisynthetic_data/poppunk/threshold/threshold_clusters.csv \
  -c /mnt/e/ecoli_db/ecoli.json \
  -ph /mnt/e/chronostrain/phylogeny/ClermonTyping/umb_phylogroups_complete.txt \
  -n 6 \
  -s "${base_seed}0" \
  -a "${RELATIVE_GROUND_TRUTH}" \
  -o "${genome_dir}" \

# ======================== genomes to simulate reads from
noise_rate=${NOISE_GENOME_MUTATION_RATE}
#db_json_replicate=${replicate_dir}/databases/chronostrain/ecoli.json
seed=0
while read line; do
  if [[ ${line:0:2} == "##" ]]; then continue; fi
  acc=$line
  seed=$((seed+1))
  create_mutant "${acc}" "${acc}.READSIM_MUTANT" "${base_seed}${seed}" $noise_rate $noise_rate ${CHRONOSTRAIN_DB_JSON_SRC}
done < "${genome_dir}/target_genomes.txt"

#create_mutant NZ_CP051001.1 NZ_CP051001.1.READSIM_MUTANT "${base_seed}4" $noise_rate $noise_rate ${}
#create_mutant NZ_CP068279.1 NZ_CP068279.1.READSIM_MUTANT "${base_seed}5" $noise_rate $noise_rate ${CHRONOSTRAIN_DB_JSON_SRC}
#create_mutant NZ_CP035882.1 NZ_CP035882.1.READSIM_MUTANT "${base_seed}6" $noise_rate $noise_rate ${CHRONOSTRAIN_DB_JSON_SRC}
#
#create_mutant NZ_CP051001.1.sim_mutant NZ_CP051001.1.sim_mutant.READSIM_MUTANT "${base_seed}7" $noise_rate $noise_rate ${CHRONOSTRAIN_DB_JSON_SRC}
#create_mutant NZ_CP068279.1.sim_mutant NZ_CP068279.1.sim_mutant.READSIM_MUTANT "${base_seed}8" $noise_rate $noise_rate ${CHRONOSTRAIN_DB_JSON_SRC}
#create_mutant NZ_CP035882.1.sim_mutant NZ_CP035882.1.sim_mutant.READSIM_MUTANT "${base_seed}9" $noise_rate $noise_rate ${CHRONOSTRAIN_DB_JSON_SRC}
