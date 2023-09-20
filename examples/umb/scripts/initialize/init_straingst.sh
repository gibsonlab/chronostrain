#!/bin/bash
set -e
source settings.sh


# Step 1: separate plasmids and archive them
seq_dir="${STRAINGE_DB_DIR}/sequences"
kmer_dir="${STRAINGE_DB_DIR}/kmers"
python prepare_strainge_db.py -i "${INDEX_REFSEQ}" -t "${seq_dir}"

# Step 2: Kmerize
#mkdir -p "${kmer_dir}"
#for f in ${seq_dir}/*.fasta; do
#  s_id=$(basename $f .fasta)
#  echo "Kmerizing ${s_id}"
#  straingst kmerize -o "${kmer_dir}/${s_id}.hdf5" "${f}"
#done

# Step 3: Cluster
cd "${STRAINGE_DB_DIR}"
#straingst kmersim --all-vs-all -t 12 -S jaccard -S subset ${kmer_dir}/*.hdf5 > similarities.tsv

# Empirical testing (Escherichia+Shigella only)
## -c 90 --> 803 clusters
## -c 92 --> 925 clusters
## -c 93 --> 986 clusters
## -c 95 --> 1147 clusters
straingst cluster -i similarities.tsv -d -C 0.99 -c 0.92 --clusters-out clusters.tsv ${kmer_dir}/*.hdf5 > references_to_keep.txt

# Step 4: pan-genome k-mer database
straingst createdb -f references_to_keep.txt -o database.hdf5
