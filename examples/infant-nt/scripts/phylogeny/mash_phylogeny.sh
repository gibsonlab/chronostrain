#!/bin/bash
set -e
source settings.sh

require_program mash


out_dir=${DATA_DIR}/database/phylogeny
mkdir -p $out_dir


echo "[*] Creating tree using Mash distances."
python phylogeny/mash_phylogeny.py \
	-j ${CHRONOSTRAIN_DB_JSON} \
	-i ${DATA_DIR}/database/ref_genomes/index.tsv \
	-o ${out_dir}/tree.nwk \
	-f 'newick'

echo "Tree created using mash_phylogeny.sh" > ${out_dir}/README.txt
echo "[*] Done."
