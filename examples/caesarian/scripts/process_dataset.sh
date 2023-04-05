#!/bin/bash
set -e
source settings.sh

participant=$1
if [ -z "$participant" ]
then
	echo "var \"participant\" is empty"
	exit 1
fi

read_dir=/mnt/e/caesarian_data/${participant}/reads
index_file=/mnt/e/caesarian_data/${participant}/chronostrain/reads.csv
> ${index_file}

append_fastq()
{
	gzip_fq_path=$1
	time=$2
	read_type=$3
	qual_fmt=$4

	num_lines=$(pigz -dc $gzip_fq_path | wc -l | awk '{print $1}')
	num_reads=$((${num_lines} / 4))

	if [[ -s "${gzip_fq_path}" ]] && [[ ${num_reads} > 0 ]]; then
		echo "Adding record ${gzip_fq_path} to ${index_file}"
		echo "${time},${num_reads},\"${gzip_fq_path}\",${read_type},${qual_fmt}" >> ${index_file}
	else
		echo "Skipping empty record ${gzip_fq_path}"
	fi
}



cd ${read_dir}
for f in *.fastq.gz; do
  basename=$(echo "${f}" | cut -d "." -f 1)
  timepoint=$(echo "${basename}" | cut -d "_" -f 1)
  sample=$(echo "${basename}" | cut -d "_" -f 2)
  pair=$(echo "${basename}" | cut -d "_" -f 3)
  if [[ "${pair}" == "2" ]]
  then
    continue
  fi

  echo "[*] -=-=-=-=-=-=-=-= Handling ${sample} (timepoint ${timepoint}). =-=-=-=-=-=-=-=-"
  # raw read files
  fq1=${timepoint}_${sample}_1.fastq.gz
  fq2=${timepoint}_${sample}_2.fastq.gz

  # Target files
  outdir=${read_dir}/${timepoint}_${sample}_kneaddata
  mkdir -p ${outdir}
  trimmed_1_paired_gz=${outdir}/${timepoint}_${sample}_paired_1.fastq.gz
  trimmed_1_unpaired_gz=${outdir}/${timepoint}_${sample}_unmatched_1.fastq.gz
  trimmed_2_paired_gz=${outdir}/${timepoint}_${sample}_paired_2.fastq.gz
  trimmed_2_unpaired_gz=${outdir}/${timepoint}_${sample}_unmatched_2.fastq.gz

  if [ -f "${trimmed_1_paired_gz}" ] && [ -f "${trimmed_2_paired_gz}" ]
	then
		echo "[*] Processed outputs already found!"
		continue
	else
    tmp_file_1="${outdir}/${timepoint}_${sample}_1.fastq"
    tmp_file_2="${outdir}/${timepoint}_${sample}_2.fastq"
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
      --trimmomatic /home/youn/miniconda3/envs/chronostrain/share/trimmomatic-0.39-2 \
      --output-prefix ${timepoint}_${sample}

    echo "[*] Compressing fastq files."
    trimmed_1_paired="${outdir}/${timepoint}_${sample}_paired_1.fastq"
    trimmed_1_unpaired="${outdir}/${timepoint}_${sample}_unmatched_1.fastq"
    trimmed_2_paired="${outdir}/${timepoint}_${sample}_paired_2.fastq"
    trimmed_2_unpaired="${outdir}/${timepoint}_${sample}_unmatched_2.fastq"
    pigz $trimmed_1_paired
    pigz $trimmed_1_unpaired
    pigz $trimmed_2_paired
    pigz $trimmed_2_unpaired

    echo "[*] Cleaning up..."
    for f in ${outdir}/*.fastq; do rm $f; done
  fi

  # Add to timeseries input index.
  append_fastq ${trimmed_1_paired_gz} $timepoint "paired_1" "fastq"
  append_fastq ${trimmed_1_unpaired_gz} $timepoint "paired_1" "fastq"
  append_fastq ${trimmed_2_paired_gz} $timepoint "paired_2" "fastq"
  append_fastq ${trimmed_2_unpaired_gz} $timepoint "paired_2" "fastq"
done
