#!/bin/bash
set -e
source settings.sh

require_program poppunk


POPPUNK_DIR=/mnt/e/infant_nt/database/poppunk_files
mkdir -p $POPPUNK_DIR
cd $POPPUNK_DIR
> input.tsv


echo "[*] Adding RefSeq entries to poppunk input."
while IFS=$'\t' read -r genus species strain accession assembly seqpath chromosomelen gffpath
do
  if [ "${species}" == "faecalis" ]; then
    echo "R_${accession}	${seqpath}" >> input.tsv
  fi
done < ${DATA_DIR}/database/ref_genomes/index.tsv


echo "[*] Adding Infant isolate assemblies to poppunk input."
python ${BASE_DIR}/helpers/list_all_participants.py ${ENA_ISOLATE_ASSEMBLY_CATALOG} | while read line
do
  participant=$line
  isolate_dir=${DATA_DIR}/${participant}/isolate_assemblies
  if [ ! -d ${isolate_dir} ]; then continue; fi
  while IFS=$'\t' read -r _participant accession fastapath genus species timepoint sampleid
  do
    if [ "${species}" == "faecalis" ]; then
      echo "A_${accession}	${fastapath}" >> input.tsv
    fi
  done < ${isolate_dir}/metadata.tsv
done


echo "[*] Running poppunk sketching (--create-db)"
poppunk --create-db --output database --r-files input.tsv --threads 8

echo "[*] Running poppunk model fit (--fit-model) with DBSCAN"
poppunk --fit-model dbscan --ref-db database --output dbscan --threads 8

echo "[*] Running poppunk model fit (--fit-model) refinement"
poppunk --fit-model refine --ref-db database --model-dir dbscan --output refine --threads 8 --max-a-dist 0.9 --max-pi-dist 0.9

echo "[*] Done."
