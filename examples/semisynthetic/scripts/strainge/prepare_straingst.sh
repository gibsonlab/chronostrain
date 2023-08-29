#!/bin/bash
set -e
source settings.sh
source strainge/settings.sh


# ================= Database using complete genomes
mkdir -p ${STRAINGST_REF_FILES}
cd ${STRAINGST_REF_FILES}


echo "[*] Kmerizing."
mkdir -p kmers
cd kmers

while IFS=$'\t' read -r -a columns
do
  seq_path="${columns[5]}"
  if [ "${seq_path}" == "SeqPath" ]; then continue; fi

  genus="${columns[0]}"
  if [ "${genus}" != 'Escherichia' ] && [ "${genus}" != 'Shigella' ]
  then
    continue
  fi

  acc="${columns[3]}"
  hdf5_path="${acc}.hdf5"
  if [ -f "${hdf5_path}" ]; then continue; fi
  straingst kmerize -o "$hdf5_path" "$seq_path"
done < "$REFSEQ_INDEX"


echo "[*] Clustering."
cd ${STRAINGST_REF_FILES}
if ! [ -f "similarities.tsv" ]; then
  straingst kmersim --all-vs-all -t ${N_CORES} -S jaccard -S subset kmers/*.hdf5 > similarities.tsv
fi
if ! [ -f "references_to_keep.txt" ]; then
  straingst cluster -i similarities.tsv -d -C 0.99 -c 0.90 --clusters-out clusters.tsv kmers/*.hdf5 > references_to_keep.txt
fi


echo "[*] Creating database."
cd ${STRAINGST_DB_DIR}/kmers
straingst createdb -f ../references_to_keep.txt -o ${STRAINGST_CHROMOSOME_DB_HDF5}

cd ${STRAINGST_REF_FILES}

for (( replicate = 1; replicate < ${N_GENOME_REPLICATES}+1; replicate++ )); do
  echo "[*] Kmerizing replicate genomes (replicate=${replicate})"
  replicate_dir=$(get_replicate_dir "${replicate}")
  straingst_db_dir=$(get_straingst_db_dir "${replicate}")
  mkdir -p "${straingst_db_dir}"
  straingst kmerize -o "${straingst_db_dir}/NZ_CP022154.1.sim_mutant.hdf5" "${replicate_dir}/sim_genomes/NZ_CP022154.1.sim_mutant.fasta"
  straingst kmerize -o "${straingst_db_dir}/NZ_LR536430.1.sim_mutant.hdf5" "${replicate_dir}/sim_genomes/NZ_LR536430.1.sim_mutant.fasta"

  echo "[*] Creating database for replicate ${replicate}."
  cat references_to_keep.txt > ${straingst_db_dir}/references_to_keep.txt
  echo "${straingst_db_dir}/NZ_CP022154.1.sim_mutant.hdf5" >> ${straingst_db_dir}/references_to_keep.txt
  echo "${straingst_db_dir}/NZ_LR536430.1.sim_mutant.hdf5" >> ${straingst_db_dir}/references_to_keep.txt
  straingst createdb -f "${straingst_db_dir}/references_to_keep.txt" -o "${straingst_db_dir}/db.hdf5"
done
