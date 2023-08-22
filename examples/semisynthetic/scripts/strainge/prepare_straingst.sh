#!/bin/bash
set -e
source settings.sh


# ================= Database using complete genomes
mkdir -p ${STRAINGST_DB_DIR}/kmers
cd ${STRAINGST_DB_DIR}/kmers

#echo "[*] Kmerizing."
#while IFS=$'\t' read -r -a columns
#do
#  seq_path="${columns[5]}"
#  if [ "${seq_path}" == "SeqPath" ]; then continue; fi
#
#  genus="${columns[0]}"
#  if [ "${genus}" != 'Escherichia' ] && [ "${genus}" != 'Shigella' ]
#  then
#    continue
#  fi
#
#  acc="${columns[3]}"
#  straingst kmerize -o ${acc}.hdf5 $seq_path
#done < $REFSEQ_INDEX


echo "[*] Clustering."
cd ${STRAINGST_DB_DIR}
#straingst kmersim --all-vs-all -t ${N_CORES} -S jaccard -S subset kmers/*.hdf5 > similarities.tsv
#straingst cluster -i similarities.tsv -d -C 0.99 -c 0.90 --clusters-out clusters.tsv kmers/*.hdf5 > /dev/null

echo "[*] Pruning target accessions."
python ${BASE_DIR}/scripts/strainge/pick_representatives.py \
  -c ${STRAINGST_DB_DIR}/clusters.tsv \
  -o ${STRAINGST_DB_DIR}/references_to_keep.txt \
  -s ${BASE_DIR}/files/ground_truth.csv

echo "[*] Creating database."
cd ${STRAINGST_DB_DIR}/kmers
straingst createdb -f ../references_to_keep.txt -o ${STRAINGST_CHROMOSOME_DB_HDF5}

echo "[*] Cleaning up."
cd ..
rm -rf ./kmers
