#!/bin/bash
set -e

source settings.sh


sample_name=$1
out_dir=${OUTPUT_DIR}/assembly/${sample_name}
cd ${out_dir}

if [ -f _BLAST_FINISHED.txt ]; then
	echo "BLAST queries for ${sample_name} finished in a previous run. Skipping this step."
	exit 0;
fi

export BLASTDB="${CHRONOSTRAIN_MARKERS_DIR}"

if [ ! -f ${CHRONOSTRAIN_MARKERS_DIR}/${BLAST_DB_NAME}.ndb ]; then
	bash helpers/create_blast_db.sh
fi

blastn \
    -db ${BLAST_DB_NAME} \
    -query spades_output/scaffolds.fasta \
    -outfmt "6 saccver sstart send slen qseqid qstart qend qlen evalue pident gaps qcovhsp" \
    -out scaffold_blastn.tsv \
    -strand both \
    -evalue 1e-50

echo "finished." > _BLAST_FINISHED.txt
