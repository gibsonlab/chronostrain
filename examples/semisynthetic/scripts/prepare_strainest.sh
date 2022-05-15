#!/bin/bash
set -e
source settings.sh

# Step 1: Align representatives to Species Representative (K-12 MG1655).
output_fasta=${STRAINEST_DB_DIR}/aln_all.fasta
strainest_script=${STRAINEST_DB_DIR}/child_script.sh
mkdir -p ${STRAINEST_DB_DIR}

seq_paths=''
export CHRONOSTRAIN_CACHE_DIR="."
python ${BASE_DIR}/helpers/strainest_helper.py \
-j ${CHRONOSTRAIN_DB_JSON} \
-i $REFSEQ_INDEX | while read strain_seq; do
	seq_paths="${seq_paths} ${strain_seq}"
done

strainest mapgenomes ${seq_paths} ${STRAIN_REP_FASTA} ${output_fasta}

# Step 2: Generate raw SNV matrix, and then cluster it.
snv_file=${STRAINEST_DB_DIR}/snvs_all.txt
strainest map2snp $STRAIN_REP_FASTA $output_fasta $snv_file

# (Note: skip clustering step.)
# Step 3: Build bowtie2 index.
cd ${STRAINEST_DB_DIR}
bowtie2-build ${output_fasta} $STRAINEST_BT2_DB
