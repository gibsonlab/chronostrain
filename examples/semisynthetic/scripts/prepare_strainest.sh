#!/bin/bash
set -e
source settings.sh


# Step 1: Pick representative genomes (to guarantee same sensitivity, use the same ones as in chronostrain)
REP_FASTAS=''
while IFS=$'\t' read -r genus species strain accession assembly seqpath; do
	if [[ "${species}" == "coli" ]]; then
		REP_FASTAS='${REP_FASTAS} ${seqpath}'
	fi
done < ${REFSEQ_INDEX}

# Step 2: Align representatives to Species Representative (K-12 MG1655).
output_fasta=${STRAINEST_DB_DIR}/aln_all.fasta
strainest mapgenomes $REP_FASTAS $STRAIN_REP_FASTA $output_fasta

# Step 3: Generate raw SNV matrix, and then cluster it.
snv_file=${STRAINEST_DB_DIR}/snvs_all.txt
snv_dist_file=${STRAINEST_DB_DIR}/snvs_dist_all.txt
snv_clust_file=${STRAINEST_DB_DIR}/snvs_clust.txt
clust_file=${STRAINEST_DB_DIR}/clusters.txt
strainest map2snp $STRAIN_REP_FASTA $output_fasta $snv_file
strainest snpdist $snv_file $snv_dist_file
strainest snpclust $snv_file $snv_dist_file $snv_clust_file $clust_file
