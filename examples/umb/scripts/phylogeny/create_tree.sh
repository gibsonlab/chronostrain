#!/bin/bash
set -e
source ../settings.sh
export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/create_tree.log"

mkdir -p ${PHYLOGENY_OUTPUT_DIR}/tree
fasttree -gtr -nt ${REFSEQ_ALIGN_PATH} > ${PHYLOGENY_OUTPUT_DIR}/tree/tree.nwk
python ${BASE_DIR}/scripts/phylogeny/tree_annotations.py \
-o ${PHYLOGENY_OUTPUT_DIR}/tree \
-p ${PHYLOGENY_OUTPUT_DIR}/ClermonTyping/umb/umb_phylogroups.txt
