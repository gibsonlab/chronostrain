#!/bin/bash
set -e
source settings.sh
source strainge/settings.sh

participant=$1
sample_id=$2
timepoint=$3
require_variable 'participant' $participant
require_variable 'sample_id' $sample_id
require_variable 'timepoint' $timepoint
require_file ${STRAINGE_DB}

# =========== Run analysis. ==================
echo "[*] Running StrainGST on participant ${participant}, sample ${sample_id} (T=${timepoint})."
run_dir=${DATA_DIR}/${participant}/strainge
mkdir -p ${run_dir}

cd $run_dir  # work locally in directory
output_file=${sample_id}.strains.tsv

if [ -f ${output_file} ]; then
  # === output already exists; skip.
  echo "[**] Output already exists. skipping."
  exit 0
else
  # === run analysis pipeline.
  echo "[**] Kmerizing."
  kmer_file=${sample_id}.hdf5
  straingst kmerize \
    -k 23 \
    -o ${kmer_file}\
    ${DATA_DIR}/${participant}/reads/${sample_id}_1.fastq.gz \
    ${DATA_DIR}/${participant}/reads/${sample_id}_2.fastq.gz

  echo "[**] Running algorithm."
  straingst run -O -o ${sample_id} ${STRAINGE_DB} ${kmer_file}

  # Validate files exist.
  if [ ! -f ${output_file} ]; then
    echo "[Error] Expected <sample_id>.strains.tsv output to exist (via -O flag). Check if StrainGST software is up to date."
    exit 1
  fi

  echo "[**] Cleaning up."
  rm $kmer_file
fi




