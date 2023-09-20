#!/bin/bash
set -e
source settings.sh

export NUM_CORES=8

export TARGET_TAXA=Klebsiella
export METAPHLAN_TAXONOMIC_KEY='g__Klebsiella'
export TARGET_DIR=./klebsiella_db  # all intermediate and final DB files will go here.

export NCBI_REFSEQ_DIR=${TARGET_DIR}/ref_genomes  # Directory to place the refseq assembly seqs.
export REFSEQ_INDEX=${TARGET_DIR}/ref_genomes/index.tsv  # The TSV index of the downloaded NCBI refseq assemblies.
export BLAST_DB_DIR=${TARGET_DIR}/blast_db  # where to store BLAST database.
export MARKER_SEED_INDEX=${TARGET_DIR}/marker_seeds/marker_seed_index.tsv  # place to store marker FASTA files
export CHRONOSTRAIN_TARGET_JSON=${TARGET_DIR}/klebsiella.json  # The final product.

export BLAST_DB_NAME=Klebsiella_refseq  # target BLAST database name.
export METAPHLAN_DB_PATH=~/miniconda3/envs/metaphlan/lib/python3.10/site-packages/metaphlan/metaphlan_databases/mpa_vJan21_CHOCOPhlAnSGB_202103.pkl  # MetaPhlan 3 or newer
export MIN_PCT_IDTY=90  # 90: "Markers" are given by 90% identity alignments.


# ========= Main body
# For an explanation, refer to the README.

if [ ! -f $METAPHLAN_DB_PATH ]
then
  echo "File ${METAPHLAN_DB_PATH} not found."
  exit 1
fi

bash download_ncbi2.sh

python python_helpers/extract_metaphlan_markers.py \
  -t $METAPHLAN_TAXONOMIC_KEY \
  -i $METAPHLAN_DB_PATH \
  -o $MARKER_SEED_INDEX

bash create_blast_db.sh

chronostrain -c chronostrain.ini \
  make-db \
  -m $MARKER_SEED_INDEX \
  -r $REFSEQ_INDEX \
  -b $BLAST_DB_NAME -bd $BLAST_DB_DIR \
  --min-pct-idty $MIN_PCT_IDTY \
  -o $CHRONOSTRAIN_TARGET_JSON
