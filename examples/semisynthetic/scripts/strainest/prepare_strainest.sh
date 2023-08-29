#!/bin/bash
set -e
source settings.sh
source strainest/settings.sh


require_file "${STRAIN_REP_FASTA}"
require_variable META_ALIGN_FASTA ${META_ALIGN_FASTA}


echo "[*] Creating database for metagenome alignment of reads."
echo "[**] Building representatives."


breadcrumb=${DATA_DIR}/StrainEst_files/bt2_database.DONE
if [ -f "${breadcrumb}" ]; then
  asdf
else
  #python strainest/metagenome_alignment_representatives.py \
  #  -i ${REFSEQ_INDEX} \
  #  -w ${DATA_DIR}/StrainEst_files/_tmp \
  #  -o ${DATA_DIR}/StrainEst_files/cluster_reps.txt \
  #  -t 12
  echo ${seq_paths}

  seq_paths=""
  num_paths=0
  while read -r seq_path; do
    seq_paths="${seq_paths} ${seq_path}"
    num_paths=$((num_paths + 1))
  done < ${DATA_DIR}/StrainEst_files/cluster_reps.txt
  seq_paths=$(echo "${seq_paths}" | xargs)  # remove trailing spaces
  echo "(Found ${num_paths} genomes for metagenome alignment DB.)"

  echo "[**] Performing multiple alignment."
  strainest mapgenomes ${seq_paths} ${STRAIN_REP_FASTA} ${META_ALIGN_FASTA}

  echo "[**] Creating Bowtie2 database."
  bowtie2-build "${META_ALIGN_FASTA}" "${META_ALIGN_BOWTIE2_DB}"

  echo "[**] Cleaning up."
  rm -rf ${DATA_DIR}/StrainEst_files/_tmp
  touch "${breadcrumb}"
fi



#echo "[*] Creating per-replicate database for SNV profiling."
#for (( replicate = 1; replicate < ${N_GENOME_REPLICATES}+1; replicate++ )); do
#  strainest_db_dir=$(get_strainest_db_dir "${replicate}")
#  seq_paths=""
#  num_paths=0
#  python strainest/snv_profiling_representatives.py | while read seq_path; do
#    seq_paths="${seq_paths} ${seq_path}"
#    num_paths=$((num_paths + 1))
#  done
#
#  echo "[**] Performing multiple alignment."
#  aln_fasta=${strainest_db_dir}/MR.fasta
#  strainest mapgenomes ${seq_paths} ${STRAIN_REP_FASTA} ${aln_fasta}
#
#  echo "[**] Mapping SNVs."
#  snv_file=${strainest_db_dir}/snp.dgrp
#  strainest map2snp ${STRAIN_REP_FASTA} ${aln_fasta} ${snv_file}
#
#  echo "[**] Computing SNV distances."
#  snv_dist_file=${strainest_db_dir}/snp_dist.txt
#  strainest snpdist ${snv_file} ${snv_dist_file}
#
#  echo "[**] Computing SNV clusters."
#  snv_file_clust=${strainest_db_dir}/snp_clust.dgrp
#  cluster_txt=${strainest_db_dir}/clusters.txt
#  strainest snpclust ${snv_file} ${snv_dist_file} ${snv_file_clust} ${cluster_txt}
#
#  echo "------------- DEBUG ensure that the target sim genomes are in seperate clusters!"
#  break
#done



  # =================== DB for SNV profiling

  # Step 1: Align representatives to Species Representative (K-12 MG1655).
#  alignment_fasta=${strainest_db_dir}/aln_all.fasta
#  mkdir -p "${strainest_db_dir}"
#
#  echo "[*] Extracting strain accessions from ${CHRONOSTRAIN_DB_JSON}"
#  while IFS=$'\t' read -r -a columns
#  do
#    seq_path="${columns[5]}"
#    if [ "${seq_path}" == "SeqPath" ]; then continue; fi
#
#    genus="${columns[0]}"
#    if [ "${genus}" != 'Escherichia' ] && [ "${genus}" != 'Shigella' ]
#    then
#      continue
#    fi
#
#    acc="${columns[3]}"
#    hdf5_path="${acc}.hdf5"
#    if [ -f "${hdf5_path}" ]; then continue; fi
#    straingst kmerize -o "$hdf5_path" "$seq_path"
#  done < "$REFSEQ_INDEX"
#
#  seq_paths=$(python ${BASE_DIR}/helpers/list_strain_paths.py --esch_shig_only -j ${CHRONOSTRAIN_DB_JSON} -i $REFSEQ_INDEX | paste -s -d " ")
#  num_paths=$(python ${BASE_DIR}/helpers/list_strain_paths.py --esch_shig_only -j ${CHRONOSTRAIN_DB_JSON} -i $REFSEQ_INDEX | wc -l)
#  echo "(Found ${num_paths} genomes.)"
#
#  echo "[*] Performing alignment..."
#  strainest mapgenomes ${seq_paths} ${STRAIN_REP_FASTA} ${alignment_fasta}
#
#  # Step 2: Generate raw SNV matrix, and then cluster it.
#  echo "[*] Creasing SNV matrix..."
#  snv_file=${STRAINEST_DB_DIR}/snvs_all.txt
#  strainest map2snp $STRAIN_REP_FASTA $alignment_fasta $snv_file
#
#  # Step 3: Clustering.
#  echo "[*] Clustering..."
#  snv_dist_file=${STRAINEST_DB_DIR}/snv_dist.txt
#  snv_clust_file=${STRAINEST_DB_DIR}/snvs_clust.txt
#  histogram=${STRAINEST_DB_DIR}/histogram.pdf
#  clusters_file=${STRAINEST_DB_DIR}/clusters.txt
#  alignment_clust_fasta=${STRAINEST_DB_DIR}/aln_clust.fasta
#  strainest snpdist ${snv_file} ${snv_dist_file} ${histogram}
#  strainest snpclust ${snv_file} ${snv_dist_file} ${snv_clust_file} ${clusters_file}
#
#  cluster_seq_paths=$(python ${BASE_DIR}/helpers/parse_strainest_clusters.py --clusters_file ${clusters_file} --refseq_index ${REFSEQ_INDEX} | paste -s -d " ")
#  strainest mapgenomes ${cluster_seq_paths} ${STRAIN_REP_FASTA} ${alignment_clust_fasta}
#
#  # Step 4: Build bowtie2 index.
#  cd ${STRAINEST_DB_DIR}
#  echo "[*] Building bowtie2 index..."
#  mkdir -p clustered
#  mkdir -p unclustered
#  bowtie2-build ${alignment_clust_fasta} clustered/$STRAINEST_BT2_DB   # NOTE: this is for clustered analysis.
