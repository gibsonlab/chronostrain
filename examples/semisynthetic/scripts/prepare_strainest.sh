#!/bin/bash
set -e
source settings.sh

# Step 1: Align representatives to Species Representative (K-12 MG1655).
alignment_fasta=${STRAINEST_DB_DIR}/aln_all.fasta
strainest_script=${STRAINEST_DB_DIR}/child_script.sh
mkdir -p ${STRAINEST_DB_DIR}

seq_paths=$(python ${BASE_DIR}/helpers/list_strain_paths.py -j ${CHRONOSTRAIN_DB_JSON} -i $REFSEQ_INDEX | paste -s -d " ")
strainest mapgenomes ${seq_paths} ${STRAIN_REP_FASTA} ${alignment_fasta}

# Step 2: Generate raw SNV matrix, and then cluster it.
snv_file=${STRAINEST_DB_DIR}/snvs_all.txt
strainest map2snp $STRAIN_REP_FASTA $alignment_fasta $snv_file

# Step 3: Clustering.
snv_dist_file=${STRAINEST_DB_DIR}/snv_dist.txt
snv_clust_file=${STRAINEST_DB_DIR}/snvs_clust.txt
histogram=${STRAINEST_DB_DIR}/histogram.pdf
clusters_file=${STRAINEST_DB_DIR}/clusters.txt
alignment_clust_fasta=${STRAINEST_DB_DIR}/aln_clust.fasta
strainest snpclust ${snv_file} ${snv_dist_file} ${histogram}
strainest snpclust ${snv_file} ${snv_dist_file} ${snv_clust_file} ${clusters_file}
strainest mapgenomes ${cluster_seq_paths} ${STRAIN_REP_FASTA} ${alignment_clust_fasta}

# Step 4: Build bowtie2 index.
cd ${STRAINEST_DB_DIR}
bowtie2-build ${alignment_clust_fasta} $STRAINEST_BT2_DB
