#!/bin/bash
set -e

source settings.sh


sample_name=$1
out_dir=${OUTPUT_DIR}/assembly/${sample_name}

if [ -f ${out_dir}/_PRIMER_SEARCH_FINISHED.txt ]; then
	echo "Primer queries for ${sample_name} finished in a previous run. Skipping this step."
	exit 0;
fi


python helpers/primer_search.py \
	--scaffold-path ${out_dir}/spades_output/scaffolds.fasta \
	--primer-path helpers/clermont_primers.fa \
	--out-path ${out_dir}/primer_hits.tsv
