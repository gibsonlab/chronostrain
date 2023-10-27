#!/bin/bash
set -e

STRAINGE_FILES_DIR=/mnt/e/infant_nt/database/strainge_files
REFSEQ_INDEX=/mnt/e/infant_nt/database/ref_genomes/index.tsv
mkdir -p ${STRAINGE_FILES_DIR}
cd ${STRAINGE_FILES_DIR}


echo "[*] Step 1 -- Kmerizing reference genomes."
mkdir -p strainge_db
while IFS=$'\t' read genus species strain accession assembly seqpath chrlen gffpath
do
	if [ "${seqpath}" == "SeqPath" ]; then continue; fi  # skip header
	if [ "${genus} ${species}" != "Enterococcus faecalis" ]; then continue; fi  # skip header
	echo "Found: ${accession} (${genus} ${species}, strain ${strain})"
  straingst kmerize -o strainge_db/${accession}.hdf5 ${seqpath}
done < ${REFSEQ_INDEX}


echo "[*] Step 2 -- Clustering."
straingst kmersim --all-vs-all -t 4 -S jaccard -S subset strainge_db/*.hdf5 > similarities.tsv
straingst cluster -i similarities.tsv -d -C 0.99 -c 0.90 \
  --clusters-out clusters.tsv \
  strainge_db/*.hdf5 > references_to_keep.txt


echo "[*] Step 3 -- Database of pan-genome k-mers."
straingst createdb -f references_to_keep.txt -o pan-genome-db.hdf5
