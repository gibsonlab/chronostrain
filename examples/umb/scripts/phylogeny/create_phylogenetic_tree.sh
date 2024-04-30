#!/bin/bash
set -e
source settings.sh


python create_phylogenetic_tree.py \
  -da ${CHRONOSTRAIN_DB_DIR}/distances.npy \
  -do ${CHRONOSTRAIN_DB_DIR}/distance_order.txt \
  -o ${PHYLOGENY_OUTPUT_DIR}/tree/tree.nwk
