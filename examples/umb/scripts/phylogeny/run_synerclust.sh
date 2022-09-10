#!/bin/bash
set -e
source settings.sh

synerclust_dir=/home/lactis/SynerClust/bin
out_dir=${PHYLOGENY_OUTPUT_DIR}/synerclust
mkdir -p $out_dir


echo "[*] Creating inputs for SynerClust."
python create_tree_for_synerclust.py \
	-s /mnt/e/strainge/references_to_keep.txt \
	-sdb /mnt/e/strainge/strainge_db \
	-j /mnt/e/chronostrain/umb_database/database_pruned_resolved.json \
	-i ${NCBI_REFSEQ_DIR}/index.tsv \
	-o ${out_dir} \
	-f 'newick'


echo "[*] Running Synerclust (PATH: ${synerclust_dir})"
python $synerclust_dir/synerclust.py \
	-w ${out_dir} \
	-r ${out_dir}/synerclust_input.txt \
	-t ${out_dir}/tree.nwk \
	--run single

echo "[*] Done."
