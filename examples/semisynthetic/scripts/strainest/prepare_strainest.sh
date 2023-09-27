#!/bin/bash
set -e
source settings.sh
source strainest/settings.sh


require_file "${SPECIES_REP_FASTA}"



# ===================== Metagenome alignment; pick ten representatives and build SNV
echo "[*] Creating database for metagenome alignment."

echo "[**] Building representatives."
python strainest/metagenome_alignment_representatives.py \
  -i "${REFSEQ_INDEX}" \
  -w "${STRAINEST_DB_DIR}/mash" \
  -o "${METAGENOME_ALIGN_DIR}/cluster_reps.txt" \
  -t 12


seq_paths=""
num_paths=0
while read -r seq_path; do
  seq_paths="${seq_paths} ${seq_path}"
  num_paths=$((num_paths + 1))
done < "${METAGENOME_ALIGN_DIR}/cluster_reps.txt"
seq_paths=$(echo "${seq_paths}" | xargs)  # remove trailing spaces
echo "(Found ${num_paths} genomes for metagenome alignment DB.)"

echo "[**] Performing multiple alignment."
mkdir -p "${METAGENOME_ALIGN_DIR}"
aln_fasta="${METAGENOME_ALIGN_DIR}/MA.fasta"
strainest mapgenomes ${seq_paths} ${SPECIES_REP_FASTA} "${aln_fasta}"


echo "[**] Building bowtie2 index."
bowtie2-build ${METAGENOME_ALIGN_DIR}/MA.fasta ${METAGENOME_ALIGN_DIR}/MA


# =================== SNV profiling; re-use poppunk clustering.
echo "[*] Creating database for SNV profiling."
echo "[**] Building representatives."
mkdir -p ${SNV_PROFILE_DIR}
python strainest/snv_profile_representatives.py \
  -i "${REFSEQ_INDEX}" \
  -p "${DATA_DIR}/poppunk/threshold/threshold_clusters.csv" \
  -w "${STRAINEST_DB_DIR}/mash" \
  -o "${SNV_PROFILE_DIR}/cluster_reps.txt" \
  -t 12

# --> create symlinks. Some strain names have punctuations in them (and in the file pathname) and breaks StrainEst.
tmpdir=${SNV_PROFILE_DIR}/tmp
mkdir -p $tmpdir
seq_paths=""
num_paths=0
while read -r full_seq_path; do
  bn="$(basename $full_seq_path)"
  seq_path=${tmpdir}/${bn}
  ln -s ${full_seq_path} ${seq_path}
  seq_paths="${seq_paths} ${seq_path}"
  num_paths=$((num_paths + 1))
done < "${SNV_PROFILE_DIR}/cluster_reps.txt"
seq_paths=$(echo "${seq_paths}" | xargs)  # remove trailing spaces
echo "(Found ${num_paths} genomes for SNV profiling.)"

echo "[**] Performing multiple alignment."
aln_fasta="${SNV_PROFILE_DIR}/MR.fasta"
strainest mapgenomes ${seq_paths} ${SPECIES_REP_FASTA} "${aln_fasta}"

echo "[**] Performing map2snp."
snv_file=${SNV_PROFILE_DIR}/snp.dgrp
strainest map2snp ${SPECIES_REP_FASTA} ${aln_fasta} ${snv_file}

echo "[**] Performing snpdist."
snv_dist_file=${SNV_PROFILE_DIR}/snp_dist.txt
hist_file=${SNV_PROFILE_DIR}/snp_hist.pdf
strainest snpdist ${snv_file} ${snv_dist_file} ${hist_file}

echo "[**] Computing SNV clusters."
snv_clust_file=${SNV_PROFILE_DIR}/snp_clust.dgrp
cluster_txt=${SNV_PROFILE_DIR}/clusters.txt
strainest snpclust ${snv_file} ${snv_dist_file} ${snv_clust_file} ${cluster_txt} -t 0.002

echo "[**] Cleaning up."
rm -rf "${tmpdir}"
