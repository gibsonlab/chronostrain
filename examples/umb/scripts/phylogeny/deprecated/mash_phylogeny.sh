#!/bin/bash
# ======== DEPRECATED
set -e
source settings.sh

out_dir=${PHYLOGENY_OUTPUT_DIR}/tree_mash
mkdir -p $out_dir


echo "[*] Creating tree using Mash distances."
python helpers/mash_phylogeny.py \
	-s /mnt/e/strainge/references_to_keep.txt \
	-sdb /mnt/e/strainge/strainge_db \
	-j /mnt/e/chronostrain/umb_database/database_pruned_resolved.json \
	-i ${NCBI_REFSEQ_DIR}/index.tsv \
	-o ${out_dir} \
	-f 'newick'

echo "[*] Done."
