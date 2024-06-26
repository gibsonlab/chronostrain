#!/bin/bash
set -e
source settings.sh

n_reads=$1
trial=$2
time_point=$3
sensitivity=$4

if [ -z "$n_reads" ]
then
	echo "var \"n_reads\" is empty"
	exit 1
fi

if [ -z "$trial" ]
then
	echo "var \"trial\" is empty"
	exit 1
fi

if [ -z "$time_point" ]
then
	echo "var \"time_point\" is empty"
	exit 1
fi

if [ "$sensitivity" != "sensitive" ] && [ "$sensitivity" != "default" ]
then
	echo "Invalid value for parameter 'sensitivity' (got: ${sensitivity}) (options: sensitive/default)."
	exit 1
fi

trial_dir=$(get_trial_dir $n_reads $trial)
read_dir=${trial_dir}/reads
output_dir=${trial_dir}/output/strainest

echo "[*] Running inference for n_reads: ${n_reads}, trial: ${trial}, timepoint #${time_point}"

mkdir -p ${output_dir}
cd ${output_dir}

reads_1="${read_dir}/${time_point}_reads_1.fq.gz"
reads_2="${read_dir}/${time_point}_reads_2.fq.gz"

# Perform bowtie2 alignment
sam_file="reads_${time_point}.sam"
bam_file="reads_${time_point}.bam"
sorted_bam_file="reads_${time_point}.sorted.bam"

export BOWTIE2_INDEXES=${STRAINEST_DB_DIR}
echo "[*] Running alignment..."
bowtie2 \
--very-fast --no-unal --quiet \
-k 2 \
-x ${STRAINEST_BOWTIE2_DB_NAME} \
-1 ${reads_1} \
-2 ${reads_2} \
-S ${sam_file}

# Invoke samtools
echo "[*] Invoking samtools..."
samtools view -b ${sam_file} > ${bam_file}
samtools sort ${bam_file} -o ${sorted_bam_file}
samtools index ${sorted_bam_file}

# Run StrainEst
echo "[*] Running StrainEst... ($sensitivity)"
if [ "$sensitivity" == "sensitive" ]; then
	strainest est ${BASE_DIR}/files/strainest_snvs.dgrp ${sorted_bam_file} ./ -p 0 -a 2
else
	strainest est ${BASE_DIR}/files/strainest_snvs.dgrp ${sorted_bam_file} ./
fi

# Clean up
echo "[*] Cleaning up..."
rm ${sam_file}
rm ${bam_file}
rm ${sorted_bam_file}

if [ -f abund.txt ]; then
	mv abund.txt abund_${time_point}.${sensitivity}.txt
fi

if [ -f info.txt ]; then
	mv info.txt info_${time_point}.${sensitivity}.txt
fi

if [ -f counts.txt ]; then
	mv counts.txt counts_${time_point}.${sensitivity}.txt
fi

if [ -f max_ident.txt ]; then
	mv max_ident.txt max_ident_${time_point}.${sensitivity}.txt
fi
