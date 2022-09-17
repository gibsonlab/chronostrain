#!/bin/bash
set -e

source settings.sh


sample_name=$1
out_dir=${OUTPUT_DIR}/assembly/${sample_name}
cd ${out_dir}

export BLASTDB="/mnt/e/chronostrain/umb_database/blast_db"
blastn \
    -db esch_chrom \
    -query spades_output/scaffolds.fasta \
    -outfmt "6 saccver sstart send slen qseqid qstart qend qlen evalue pident gaps qcovhsp" \
    -out scaffold_blastn.tsv \
    -strand both \
    -evalue 1e-50
