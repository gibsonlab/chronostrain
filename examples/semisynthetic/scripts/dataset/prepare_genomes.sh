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


# ======================== Pick random genomes to simulate reads from.
python dataset/pick_random_genomes.py \
  -i ${REFSEQ_INDEX} \
  -p /mnt/e/semisynthetic_data/poppunk/threshold/threshold_clusters.csv \
  -c ${CHRONOSTRAIN_CLUSTER_FILE} \
  -ph /mnt/e/chronostrain/phylogeny/ClermonTyping/umb_phylogroups_complete.txt \
  -n 6 \
  -s "${base_seed}0" \
  -a "${RELATIVE_GROUND_TRUTH}" \
  -o "${genome_dir}" \

# ======================== genomes to simulate reads from
# multiply two floats in bash
noise_rate=$(echo $mutation_ratio $NOISE_GENOME_MUTATION_RATE | awk '{printf "%4.3f\n",$1*$2}')
echo "[*] Using mutation rate = ${noise_rate}"

#noise_rate=${NOISE_GENOME_MUTATION_RATE}
#db_json_replicate=${replicate_dir}/databases/chronostrain/ecoli.json
seed=0
while read line; do
  if [[ ${line:0:2} == "##" ]]; then continue; fi
  acc=$line
  seed=$((seed+1))
  create_mutant "${acc}" "${acc}.READSIM_MUTANT" "${base_seed}${seed}" $noise_rate $noise_rate ${CHRONOSTRAIN_DB_JSON_SRC}
done < "${genome_dir}/target_genomes.txt"
