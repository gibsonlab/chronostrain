#!/bin/bash
set -e
source settings.sh

n_reads=$1
trial=$2
time_point=$3

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
output_dir=${trial_dir}/output/constrains
cfg_file=${output_dir}/constrains_input.cfg

echo "[*] Running ConStrains for n_reads: ${n_reads}, trial: ${trial}, timepoint #${time_point}"

> $cfg_file
for time_point in 0 1 2 3 4; do
	echo "//" >> $cfg_file
	echo "sample: [${time_point}_reads]" >> $cfg_file
	echo "fq1: [${read_dir}/${time_point}_reads_1.fq.gz]" >> $cfg_file
	echo "fq2: [${read_dir}/${time_point}_reads_2.fq.gz]" >> $cfg_file
done


export PATH=${METAPHLAN2_DIR}:$PATH
export mpa_dir=${METAPHLAN2_DIR}:$PATH
python ${CONSTRAINS_DIR}/ConStrains.py \
-o ${output_dir} \
-c ${cfg_file} \
-t 4 \
--min-cov=2
