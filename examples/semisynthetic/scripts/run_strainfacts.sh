#!/bin/bash
set -e
source settings.sh

n_reads=$1
trial=$2

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

trial_dir=$(get_trial_dir $n_reads $trial)
read_dir=${trial_dir}/reads
output_dir=${trial_dir}/output/strainfacts

echo "[*] Running StrainFacts inference for n_reads: ${n_reads}, trial: ${trial}"

mkdir -p ${output_dir}
cd ${output_dir}


# Run metagenotyping
reads_1="${read_dir}/${time_point}_reads_1.fq.gz"
reads_2="${read_dir}/${time_point}_reads_2.fq.gz"
$GT_PRO_BIN genotype \
-d ${GT_PRO_DB} \
${read_dir}/0_reads_1.fq.gz ${read_dir}/0_reads_2.fq.gz \
${read_dir}/1_reads_1.fq.gz ${read_dir}/1_reads_2.fq.gz \
${read_dir}/2_reads_1.fq.gz ${read_dir}/2_reads_2.fq.gz \
${read_dir}/3_reads_1.fq.gz ${read_dir}/3_reads_2.fq.gz \
${read_dir}/4_reads_1.fq.gz ${read_dir}/4_reads_2.fq.gz \
-t ${N_CORES} \
-o all_reads
