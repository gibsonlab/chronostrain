#!/bin/bash
set -e
source settings.sh
export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/create_tree.log"

mkdir -p ${PHYLOGENY_OUTPUT_DIR}/tree


for choice in "all" "metaphlan" "mlst" "clermont"; do
	marker_multi_align=${MULTI_ALIGN_DIR}/${choice}_markers.fasta

	python ${BASE_DIR}/helpers/concatenated_multiple_alignments.py \
			--raw_json ${CHRONOSTRAIN_DB_JSON_ALL} \
			--align_path ${marker_multi_align} \
			--marker_choice "${choice}" \
			--uniprot_csv ${BASE_DIR}/files/uniprot_markers.tsv \
			--genes_fasta ${BASE_DIR}/files/clermont_genes.fasta \
			--metaphlan_db ${METAPHLAN_PKL_PATH}

	tree_output_dir=${PHYLOGENY_OUTPUT_DIR}/tree_${choice}
	fasttree -gtr -nt ${marker_multi_align} > ${tree_output_dir}/tree.nwk

	python ${BASE_DIR}/scripts/phylogeny/tree_annotations.py \
			-o ${tree_output_dir}/tree \
			-i ${NCBI_REFSEQ_DIR}/index.tsv \
			-p ${PHYLOGENY_OUTPUT_DIR}/ClermonTyping/umb_phylogroups_complete.txt
done
