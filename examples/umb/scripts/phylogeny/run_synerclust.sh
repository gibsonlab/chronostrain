#!/bin/bash
set -e
source settings.sh

synerclust_dir=/home/lactis/SynerClust/bin
sim_file=/mnt/e/strainge/similarities.tsv

out_dir=${PHYLOGENY_OUTPUT_DIR}/synerclust
mkdir -p $out_dir

tree_path=${out_dir}/newick_jaccard.nwk  # precomputed

echo "[*] Preprocessing inputs for SynerClust."
python create_tree_for_synerclust.py \
	-i ${NCBI_REFSEQ_DIR}/index.tsv \
	-o ${tree_path} \
	-f 'newick' \
	-s ${sim_file}

python create_synerclust_input.py \
	-i ${NCBI_REFSEQ_DIR}/index.tsv \
	-o ${out_dir}/input.txt


echo "[*] Running Synerclust (PATH: ${synerclust_dir})"

python $synerclust_dir/synerclust.py \
	-r ${out_dir}/input.txt \
	-w ${out_dir} \
	-t ${tree_path}


echo "[*] Generated phylogroup path ${final_path}."
