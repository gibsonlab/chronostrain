#!/bin/bash
# ===== DEPRECATED
set -e
source settings.sh
export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/create_tree.log"

db_json=$CHRONOSTRAIN_DB_JSON_ALL
for gene_choice in "all" "metaphlan3" "mlst" "clermont"; do
	echo "[*] Starting pipeline for generating phylogenetic tree (marker choice: \"${gene_choice}\")"
	marker_multi_align=${MULTI_ALIGN_DIR}/${gene_choice}_markers.fasta

	if [ -f $marker_multi_align ]; then
		echo "[*] Multiple alignment ${marker_multi_align} already found; Skipping this step."
	else
		echo "[*] Obtaining multiple alignments."

		if [[ "${gene_choice}" == "clermont" ]]; then
			python ${BASE_DIR}/helpers/concatenated_multiple_alignments.py \
					--raw_json ${CHRONOSTRAIN_DB_JSON_ALL} \
					--align_path ${marker_multi_align} \
					--marker_choice "${gene_choice}" \
					--uniprot_csv ${BASE_DIR}/files/uniprot_markers.tsv \
					--clermont_fasta ${BASE_DIR}/files/clermont_genes.fasta \
					--metaphlan3_pkl ${METAPHLAN3_PKL_PATH} \
					--force_include "trpB"
		else
			python ${BASE_DIR}/helpers/concatenated_multiple_alignments.py \
					--raw_json ${CHRONOSTRAIN_DB_JSON_ALL} \
					--align_path ${marker_multi_align} \
					--marker_choice "${gene_choice}" \
					--uniprot_csv ${BASE_DIR}/files/uniprot_markers.tsv \
					--clermont_fasta ${BASE_DIR}/files/clermont_genes.fasta \
					--metaphlan3_pkl ${METAPHLAN3_PKL_PATH}
		fi
	fi

	tree_output_dir=${PHYLOGENY_OUTPUT_DIR}/tree_${gene_choice}
	mkdir -p ${tree_output_dir}

	echo "[*] Running fasttree.."
	fasttree -gtr -nt ${marker_multi_align} > ${tree_output_dir}/tree.nwk

	echo "[*] Generating tree annotations."
	python ${BASE_DIR}/scripts/phylogeny/tree_annotations.py \
			-o ${tree_output_dir} \
			-i ${NCBI_REFSEQ_DIR}/index.tsv \
			-p ${PHYLOGENY_OUTPUT_DIR}/ClermonTyping/umb_phylogroups_complete.txt
done
