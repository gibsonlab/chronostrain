#!/bin/bash
set -e
source settings.sh


python helpers/prepare_karp.py \
-i ${NCBI_REFSEQ_DIR}/index.tsv \
-s ${STRAINGE_DB_DIR}/references_to_keep.tsv \
-m ${STRAINGE_DB_DIR}/strainge_db/references_meta.tsv \
-o ${KARP_DB_DIR}

samtools faidx ${KARP_DB_DIR}/references.fasta
karp -c index -r ${KARP_DB_DIR}/references.fasta -i ${KARP_DB_DIR}/references.index
