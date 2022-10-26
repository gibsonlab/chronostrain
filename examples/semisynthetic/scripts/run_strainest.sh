#!/bin/bash
set -e
source settings.sh


check_program bowtie2
check_program samtools
check_program strainest
check_program pigz


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
runtime_file=${trial_dir}/output/strainest_runtime.${sensitivity}.${time_point}.txt

mkdir -p ${output_dir}
cd ${output_dir}


if [[ -f $runtime_file ]]; then
	echo "[*] Skipping StrainEst run (n_reads: ${n_reads}, trial: ${trial}, timepoint #${time_point})"
	exit 0
fi


# ========== Run
echo "[*] Running inference for n_reads: ${n_reads}, trial: ${trial}, timepoint #${time_point}"
start_time=$(date +%s%N)  # nanoseconds

# Perform bowtie2 alignment
sam_file="reads_${time_point}.sam"
bam_file="reads_${time_point}.bam"
sorted_bam_file="reads_${time_point}.sorted.bam"

echo "[*] Running alignment..."
bowtie2 \
--very-fast --no-unal --quiet \
-p ${N_CORES} \
-x ${STRAINEST_DB_DIR}/unclustered/${STRAINEST_BT2_DB} \
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
echo "[*] Running StrainEst... ($sensitivity)"
if [ "$sensitivity" == "sensitive" ]; then
	strainest est \
	${STRAINEST_DB_DIR}/snvs_all.txt \
	${sorted_bam_file} \
	./ \
	-t ${N_CORES} \
	-p 0 \
	-a
else
	strainest est \
	${STRAINEST_DB_DIR}/snvs_all.txt \
	${sorted_bam_file} \
	./ \
	-t ${N_CORES}
fi

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
