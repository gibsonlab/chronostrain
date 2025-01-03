#!/bin/bash
set -e
source settings.sh
source strainge/settings.sh


# Step 1: separate plasmids and archive them
echo "[*] Preparing reference catalog."
seq_dir="${STRAINGE_DB_DIR}/sequences"
kmer_dir="${STRAINGE_DB_DIR}/kmers"
python prepare_strainge_db.py -i "${INDEX_REFSEQ}" -t "${seq_dir}"

## Step 2: Kmerize
echo "[*] Preparing k-mers."
mkdir -p "${kmer_dir}"
for f in ${seq_dir}/*.fasta; do
  s_id=$(basename $f .fasta)
  echo "Kmerizing ${s_id}"
  straingst kmerize -o "${kmer_dir}/${s_id}.hdf5" "${f}"
done

## Step 3: Cluster
echo "[*] Computing similarities."
cd "${STRAINGE_DB_DIR}"
straingst kmersim --all-vs-all -t 12 -S jaccard -S subset ${kmer_dir}/*.hdf5 > similarities.tsv

# Empirical testing (Escherichia+Shigella only)
echo "[*] Computing clusters."
## -c 0.90 --> 803 clusters, 284 phylogroup A clusters
## -c 0.907 --> 838 clusters, 293 phylogroup A clusters
## -c 0.909 --> 858 clusters, 295 phylogroup A clusters
## -c 0.91 --> 864 clusters, 298 phylogroup A clusters
## -c 0.912 --> 875 clusters, 298 phylogroup A clusters
## -c 0.913 --> 878 clusters, 299 phylogroup A clusters
## -c 0.918 --> 914 clusters, 307 phylogroup A clusters
## -c 0.92 --> 927 clusters, 309 phylogroup A clusters
## -c 0.94 --> 1065 clusters, 330 phylogroup A clusters
## -c 0.947 --> 1119 clusters, 340 phylogroup A clusters [We use this one]
## -c 0.95 --> 1147 clusters, 345 phylogroup A clusters
straingst cluster -i similarities.tsv -d -C 0.99 -c 0.947 --clusters-out clusters.tsv ${kmer_dir}/*.hdf5 > references_to_keep.txt

# Step 4: pan-genome k-mer database
echo "[*] Finalizing database."
straingst createdb -f references_to_keep.txt -o database.hdf5

echo "[*] Done."
