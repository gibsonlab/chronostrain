#!/bin/bash
set -e

source settings.sh


sample_name=$1
read1="${SAMPLES_DIR}/${sample_name}-R1.fastq.gz"
read2="${SAMPLES_DIR}/${sample_name}-R2.fastq.gz"

out_dir=${OUTPUT_DIR}/assembly/${sample_name}
mkdir -p ${out_dir}
cd ${out_dir}

# ============== Check to see if we need to run trimomomatic.
r1="${sample_name}_paired_1.fastq.gz"
r2="${sample_name}_paired_2.fastq.gz"

if [ ! -f ${r1} ]; then
    tmp_1="${sample_name}_1.fastq"
    tmp_2="${sample_name}_2.fastq"
    echo "[*] Decompressing to temporary output."
    pigz -dck $read1 > $tmp_1
    pigz -dck $read2 > $tmp_2

    echo "[*] Invoking kneaddata."
    kneaddata \
	--input1 $tmp_1 --input2 $tmp_2 \
	--reference-db /mnt/e/kneaddata_db \
	--output . \
	--trimmomatic-options "ILLUMINACLIP:${NEXTERA_ADAPTER_PATH}:2:30:10:2 LEADING:10 TRAILING:10 MINLEN:35" \
	--threads 8 \
	--quality-scores phred33 \
	--bypass-trf \
	--trimmomatic /home/lactis/anaconda3/envs/chronostrain/share/trimmomatic-0.39-2 \
	--output-prefix $sample_name

    echo "[*] Compressing fastq files."
    pigz ${sample_name}_paired_1.fastq
    pigz ${sample_name}_paired_2.fastq

    echo "[*] Cleaning up."
    rm $tmp_1
    rm $tmp_2
fi


${SPADES_DIR}/spades.py \
    --meta \
    -o ./spades_output \
    -1 $r1 -2 $r2 \
    -t 8 \
    -m 60
