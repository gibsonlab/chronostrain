#!/bin/bash
set -e
source settings.sh

require_program pigz
require_program kneaddata

participant=$1
require_variable 'participant' $participant

patient_index_file=${DATA_DIR}/${participant}/dataset.tsv
require_file $patient_index_file

read_dir=${DATA_DIR}/${participant}/reads
require_dir $read_dir

mkdir -p ${DATA_DIR}/${participant}/chronostrain
chronostrain_index_file=${DATA_DIR}/${participant}/chronostrain/reads.csv
> ${chronostrain_index_file}

append_fastq()
{
	gzip_fq_path=$1
	time=$2
	read_type=$3
	qual_fmt=$4

	num_lines=$(pigz -dc $gzip_fq_path | wc -l | awk '{print $1}')
	num_reads=$((${num_lines} / 4))

	if [[ -s "${gzip_fq_path}" ]] && [[ ${num_reads} > 0 ]]; then
		echo "Adding record ${gzip_fq_path} to ${chronostrain_index_file}"
		echo "${time},${num_reads},\"${gzip_fq_path}\",${read_type},${qual_fmt}" >> ${chronostrain_index_file}
	else
		echo "Skipping empty record ${gzip_fq_path}"
	fi
}

{
  read
  while IFS=$'\t' read -r participant time_point sample_id fq1 fq2
  do
    echo "[*] -=-=-=-=-=-=-=-= Handling ${sample_id} (timepoint ${time_point}). =-=-=-=-=-=-=-=-"
    # Target files
    outdir=${read_dir}/${sample_id}_kneaddata
    mkdir -p ${outdir}

    target_prefix="${sample_id}"
    trimmed_1_paired_gz=${outdir}/${target_prefix}_paired_1.fastq.gz
    trimmed_1_unpaired_gz=${outdir}/${target_prefix}_unmatched_1.fastq.gz
    trimmed_2_paired_gz=${outdir}/${target_prefix}_paired_2.fastq.gz
    trimmed_2_unpaired_gz=${outdir}/${target_prefix}_unmatched_2.fastq.gz

    if [ -f "${trimmed_1_paired_gz}" ] && [ -f "${trimmed_2_paired_gz}" ]
    then
      echo "[*] Processed outputs already found!"
      continue
    else
      tmp_file_1="${outdir}/${target_prefix}_1.fastq"
      tmp_file_2="${outdir}/${target_prefix}_2.fastq"
      echo "[*] Decompressing to temporary output."
      pigz -dck ${fq1} > $tmp_file_1
      pigz -dck ${fq2} > $tmp_file_2

      kneaddata \
        --input1 $tmp_file_1 \
        --input2 $tmp_file_2 \
        --reference-db /mnt/e/kneaddata_db \
        --output ${outdir} \
        --trimmomatic-options "SLIDINGWINDOW:4:20 MINLEN:87" \
        --threads 8 \
        --quality-scores phred33 \
        --bypass-trf \
        --trimmomatic ${TRIMMOMATIC_PATH} \
        --output-prefix ${target_prefix}

      echo "[*] Compressing fastq files."
      trimmed_1_paired="${outdir}/${target_prefix}_paired_1.fastq"
      trimmed_1_unpaired="${outdir}/${target_prefix}_unmatched_1.fastq"
      trimmed_2_paired="${outdir}/${target_prefix}_paired_2.fastq"
      trimmed_2_unpaired="${outdir}/${target_prefix}_unmatched_2.fastq"
      pigz $trimmed_1_paired
      pigz $trimmed_1_unpaired
      pigz $trimmed_2_paired
      pigz $trimmed_2_unpaired

      echo "[*] Cleaning up..."
      for f in ${outdir}/*.fastq; do rm $f; done
    fi

    # Add to timeseries input index.
    append_fastq ${trimmed_1_paired_gz} $time_point "paired_1" "fastq"
    append_fastq ${trimmed_1_unpaired_gz} $time_point "paired_1" "fastq"
    append_fastq ${trimmed_2_paired_gz} $time_point "paired_2" "fastq"
    append_fastq ${trimmed_2_unpaired_gz} $time_point "paired_2" "fastq"
  done
} < ${patient_index_file}

touch ${DATA_DIR}/${participant}/chronostrain/process_reads.DONE