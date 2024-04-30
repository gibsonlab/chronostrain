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
runtime_file=${output_dir}/runtime.${time_point}.txt
if [[ -f $runtime_file ]]; then
	echo "[*] Skipping StrainEst run (replicate: ${replicate} | n_reads: ${n_reads} | trial: ${trial} | timepoint #${time_point})"
	exit 0
fi

echo "[*] Running inference for (replicate: ${replicate} | n_reads: ${n_reads} | trial: ${trial} | timepoint #${time_point})"
start_time=$(date +%s%N)  # nanoseconds

# Perform bowtie2 alignment
sam_file_sim="reads_${time_point}_sim.sam"
sam_file_bg="reads_${time_point}_bg.sam"

bam_file_sim="reads_${time_point}_sim.bam"
bam_file_bg="reads_${time_point}_bg.bam"
bam_file="reads_${time_point}.bam"
sorted_bam_file="reads_${time_point}.sorted.bam"

echo "[*] Running alignment..."
bowtie2 \
--very-fast --no-unal --quiet \
-p ${N_CORES} \
-x ${METAGENOME_ALIGN_DIR}/MA \
-1 ${read_dir}/${time_point}_sim_1.fq \
-2 ${read_dir}/${time_point}_sim_2.fq \
-S ${sam_file_sim}

bowtie2 \
--very-fast --no-unal --quiet \
-p ${N_CORES} \
-x ${METAGENOME_ALIGN_DIR}/MA \
-1 ${BACKGROUND_FASTQ_DIR}/sorted/${time_point}_background_1.sorted.fq \
-2 ${BACKGROUND_FASTQ_DIR}/sorted/${time_point}_background_2.sorted.fq \
-S ${sam_file_bg}

# Invoke samtools
echo "[*] Invoking samtools..."
rm -f ${bam_file_sim}
rm -f ${bam_file_bg}
rm -f ${bam_file}
rm -f ${sorted_bam_file}
samtools view -b ${sam_file_sim} > ${bam_file_sim}
samtools view -b ${sam_file_bg} > ${bam_file_bg}
samtools merge ${bam_file} ${bam_file_sim} ${bam_file_bg}
samtools sort ${bam_file} -o ${sorted_bam_file}
samtools index ${sorted_bam_file}

# Run StrainEst
echo "[*] Running StrainEst."
strainest est \
  ${SNV_PROFILE_DIR}/snp.dgrp \
  ${sorted_bam_file} \
  ./ \
  -t ${N_CORES} \
	-p 5 \
	-a 3

# ====== Record runtime
end_time=$(date +%s%N)
elapsed_time=$(( $(($end_time-$start_time)) / 1000000 ))
echo "${elapsed_time}" > $runtime_file

# ========== Clean up
echo "[*] Cleaning up..."
rm ./*.sam
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
