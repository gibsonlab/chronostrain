#!/bin/bash
set -e
source settings.sh


mkdir -p ${DATA_DIR}/sim_genomes

# Copy existing reference genomes
while IFS=$'\t' read -r -a columns
do
  acc="${columns[3]}"
  seq_path="${columns[5]}"
  if [ "${acc}" == "NZ_CP092452.1" ] || [ "${acc}" == "NZ_CP024859.1" ]; then
    cp ${seq_path} ${DATA_DIR}/sim_genomes/${acc}.fasta
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
    -i ${DATA_DIR}/sim_genomes/${src_acc}.fasta \
    -o ${DATA_DIR}/sim_genomes/${tgt_acc}.fasta \
    -j ${db_json} \
    -d ${mutation_rate} \
    -sid ${src_acc} \
    -tid ${tgt_acc} \
    -s ${seed}
}


# ======================== genomes to include in database
create_mutant NZ_CP092452.1 NZ_CP092452.1.sim_mutant 31415 0.002 ${CHRONOSTRAIN_DB_JSON_SRC}
create_mutant NZ_CP024859.1 NZ_CP024859.1.sim_mutant 27182 0.002 ${CHRONOSTRAIN_DB_JSON_SRC}


# ======================== Update chronostrain JSON.
bash chronostrain/prepare_chronostrain.sh  # this initializes the JSON file at CHRONOSTRAIN_DB_JSON


# ======================== genomes to simulate reads from
create_mutant NZ_CP092452.1 NZ_CP092452.1.READSIM_MUTANT 1 0.001 ${CHRONOSTRAIN_DB_JSON}
create_mutant NZ_CP024859.1 NZ_CP024859.1.READSIM_MUTANT 2 0.001 ${CHRONOSTRAIN_DB_JSON}
create_mutant NZ_CP092452.1.sim_mutant NZ_CP092452.1.sim_mutant.READSIM_MUTANT 3 0.001 ${CHRONOSTRAIN_DB_JSON}
create_mutant NZ_CP024859.1.sim_mutant NZ_CP024859.1.sim_mutant.READSIM_MUTANT 4 0.001 ${CHRONOSTRAIN_DB_JSON}
