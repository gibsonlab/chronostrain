#!/bin/bash
set -e
source settings.sh
source strainest/settings.sh


require_program bowtie2
require_program samtools
require_program strainest
require_program pigz


mutation_ratio=$1
replicate=$2
n_reads=$3
trial=$4
time_point=$5

require_variable "mutation_ratio" $mutation_ratio
require_variable "replicate" $replicate
require_variable "n_reads" $n_reads
require_variable "trial" $trial
require_variable "time_point" $time_point


trial_dir=$(get_trial_dir "${mutation_ratio}" "$replicate" "$n_reads" "$trial")
read_dir=${trial_dir}/reads
output_dir=${trial_dir}/output/strainest

mkdir -p ${output_dir}
cd ${output_dir}


# ======= Run
runtime_file=${trial_dir}/output/strainest_runtime.${time_point}.txt
if [[ -f $runtime_file ]]; then
	echo "[*] Skipping StrainEst run (replicate: ${replicate} | n_reads: ${n_reads} | trial: ${trial} | timepoint #${time_point})"
	exit 0
fi

echo "[*] Running inference for (replicate: ${replicate} | n_reads: ${n_reads} | trial: ${trial} | timepoint #${time_point})"
start_time=$(date +%s%N)  # nanoseconds

# Perform bowtie2 alignment
sam_file="reads_${time_point}.sam"
bam_file="reads_${time_point}.bam"
sorted_bam_file="reads_${time_point}.sorted.bam"

echo "[*] Running alignment..."
bowtie2 \
--very-fast --no-unal --quiet \
-p ${N_CORES} \
-x ${METAGENOME_ALIGN_DIR}/MA \
-U ${read_dir}/${time_point}_sim_1.fq \
-U ${read_dir}/${time_point}_sim_2.fq \
-U ${BACKGROUND_FASTQ_DIR}/${time_point}_background_1.fq \
-U ${BACKGROUND_FASTQ_DIR}/${time_point}_background_2.fq \
-S ${sam_file}

# Invoke samtools
echo "[*] Invoking samtools..."
samtools view -b ${sam_file} > ${bam_file}
samtools sort ${bam_file} -o ${sorted_bam_file}
samtools index ${sorted_bam_file}

# Run StrainEst
echo "[*] Running StrainEst."
strainest est \
  ${SNV_PROFILE_DIR}/snp.dgrp \
  ${sorted_bam_file} \
  ./ \
  -t ${N_CORES} \
	-p 1 \
	-a 1

# ====== Record runtime
end_time=$(date +%s%N)
elapsed_time=$(( $(($end_time-$start_time)) / 1000000 ))
echo "${elapsed_time}" > $runtime_file

# ========== Clean up
echo "[*] Cleaning up..."
rm ${sam_file}
rm ${bam_file}
rm ${sorted_bam_file}

if [ -f abund.txt ]; then
	mv abund.txt abund_${time_point}.txt
fi

if [ -f info.txt ]; then
	mv info.txt info_${time_point}.txt
fi

if [ -f counts.txt ]; then
	mv counts.txt counts_${time_point}.txt
fi

if [ -f max_ident.txt ]; then
	mv max_ident.txt max_ident_${time_point}.txt
fi
