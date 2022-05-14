#!/bin/bash
set -e
source settings.sh

# Step 1: Align representatives to Species Representative (K-12 MG1655).
output_fasta=${STRAINEST_DB_DIR}/aln_all.fasta
strainest_script=${STRAINEST_DB_DIR}/child_script.sh
python ${BASE_DIR}/helpers/strainest_helper.py \
-i $REFSEQ_INDEX \
-o $output_fasta \
-r $STRAIN_REP_FASTA \
-t $strainest_script

bash $strainest_script

# Step 2: Generate raw SNV matrix, and then cluster it.
snv_file=${STRAINEST_DB_DIR}/snvs_all.txt
snv_dist_file=${STRAINEST_DB_DIR}/snvs_dist_all.txt
snv_clust_file=${STRAINEST_DB_DIR}/snvs_clust.txt
clust_file=${STRAINEST_DB_DIR}/clusters.txt
strainest map2snp $STRAIN_REP_FASTA $output_fasta $snv_file
strainest snpdist $snv_file $snv_dist_file
strainest snpclust $snv_file $snv_dist_file $snv_clust_file $clust_file
