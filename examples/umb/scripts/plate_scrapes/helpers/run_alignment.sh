#!/bin/bash
set -e

source settings.sh


sample_name=$1
out_dir=${OUTPUT_DIR}/assembly/${sample_name}

if [ -f ${out_dir}/_ALIGNMENT_FINISHED.txt ]; then
	echo "Alignments for ${sample_name} finished in a previous run. Skipping this step."
	exit 0;
fi

ref_path=${CHRONOSTRAIN_MARKERS_DIR}/${MARKER_FASTA}
scaffold_path=${out_dir}/spades_output/scaffolds.fasta

echo "Target reference: ${ref_path}"
echo "Target query: ${scaffold_path}"

bwa index ${ref_path}
bwa mem \
		-o ${out_dir}/alignment.sam \
		-t 6 \
		-k 10 \
		-r 1.0 \
		-a \
		$ref_path $scaffold_path

echo "finished." > ${out_dir}/_ALIGNMENT_FINISHED.txt
