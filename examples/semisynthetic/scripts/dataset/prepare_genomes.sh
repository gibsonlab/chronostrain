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


python ${BASE_DIR}/helpers/mutate_genome.py \
  -i ${DATA_DIR}/sim_genomes/NZ_CP092452.1.fasta \
  -o ${DATA_DIR}/sim_genomes/NZ_CP092452.1.sim_mutant.fasta \
  -j ${CHRONOSTRAIN_DB_JSON_SRC} \
  -d 0.002 \
  -sid NZ_CP092452.1 \
  -tid NZ_CP092452.1.sim_mutant \
  -s 31415


python ${BASE_DIR}/helpers/mutate_genome.py \
  -i ${DATA_DIR}/sim_genomes/NZ_CP024859.1.fasta \
  -o ${DATA_DIR}/sim_genomes/NZ_CP024859.1.sim_mutant.fasta \
  -j ${CHRONOSTRAIN_DB_JSON_SRC} \
  -d 0.002 \
  -sid NZ_CP024859.1 \
  -tid NZ_CP024859.1.sim_mutant \
  -s 27182
