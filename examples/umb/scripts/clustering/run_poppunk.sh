#!/bin/bash
set -e
source settings.sh



POPPUNK_DIR=${CHRONOSTRAIN_DB_DIR}/poppunk
mkdir -p $POPPUNK_DIR
cd $POPPUNK_DIR
> input.tsv


echo "[*] Adding RefSeq entries to poppunk input."
echo "[*] Linking RefSeq catalog..."
while IFS=$'\t' read -r -a row; do
  acc="${row[3]}"
  species="${row[1]}"
  seq_path="${row[5]}"
  if [ "${species}" == "coli" ]; then
    echo "${acc}	${seq_path}" >> input.tsv
  fi
done < ${REFSEQ_INDEX}


echo "[*] Running poppunk sketching (--create-db)"
poppunk --create-db --output database --r-files input.tsv --threads 8

echo "[*] Running poppunk model fit (--fit-model) with DBSCAN"
poppunk --fit-model dbscan --ref-db database --output dbscan --threads 8

echo "[*] Running poppunk model fit (--fit-model) refinement"
poppunk --fit-model refine --ref-db database --model-dir dbscan --output refine --threads 8 --max-a-dist 0.9 --max-pi-dist 0.9

echo "[*] Done."
