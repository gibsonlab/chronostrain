#!/bin/bash
set -e
source settings.sh

synerclust_dir=/home/lactis/SynerClust/bin
tree_path=/mnt/e/strainge/straingst_umb/tree/newick_jaccard.nwk  # precomputed

echo "[*] Using SynerClust script at: ${synerclust_dir}"

python create_synerclust_input.py \
	-i ${NCBI_REFSEQ_DIR}/index.tsv \
	-o ${PHYLOGENY_OUTPUT_DIR}/synerclust/input.txt

python $synerclust_dir/synerclust.py \
	-r ${PHYLOGENY_OUTPUT_DIR}/synerclust/input.txt \
	-w ${PHYLOGENY_OUTPUT_DIR}/synerclust \
	-t $tree_path


echo "[*] Generated phylogroup path ${final_path}."
