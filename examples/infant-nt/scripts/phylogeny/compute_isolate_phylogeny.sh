#!/bin/bash
set -e
source settings.sh


require_program 'blastn'
require_file $MARKER_SEED_INDEX
require_dir $MARKER_ALIGNMENT_DIR

outdir=${DATA_DIR}/phylogeny

echo "[*] Parsing BLAST results to compute distances and phylogeny."
export CHRONOSTRAIN_CACHE_DIR=.
export CHRONOSTRAIN_LOG_FILEPATH=${outdir}/run.log
export CHRONOSTRAIN_DB_JSON="${DATA_DIR}/database/efaecalis-1raw.json"
export CHRONOSTRAIN_DB_NAME="efaecalis-1raw"

# Signal JAX that we won't need the GPU
env JAX_PLATFORM_NAME=cpu python phylogeny/compute_multiple_alignments.py \
  --participants ${BASE_DIR}/files/all_participants.txt \
  --data-dir ${DATA_DIR} \
  --out $outdir/all_multiple_alignments.fa \
  --marker-seeds $MARKER_SEED_INDEX \
  --seed-multiple-alignments $MARKER_ALIGNMENT_DIR \
  --threads 8


echo "[*] Running fasttree.."
fasttree -gtr -nt $outdir/all_multiple_alignments.fa > $outdir/tree.nwk
