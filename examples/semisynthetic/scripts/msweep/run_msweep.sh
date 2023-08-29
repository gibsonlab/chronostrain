#!/bin/bash
set -e
source settings.sh
source msweep/settings.sh

require_program mSWEEP

# ============ Requires arguments:
replicate=$1
n_reads=$2
trial=$3
time_point=$4

require_variable "replicate" $replicate
require_variable "n_reads" $n_reads
require_variable "trial" $trial
require_variable "time_point" $time_point

themisto_db_dir=$(get_themisto_db_dir "${replicate}")
trial_dir=$(get_trial_dir $replicate $n_reads $trial)
read_dir=${trial_dir}/reads
output_dir=${trial_dir}/output/msweep
pseudoalignment_dir=${trial_dir}/output/themisto
runtime_file=${trial_dir}/output/msweep_runtime.${time_point}.txt
mkdir -p $output_dir


if [ -f $runtime_file ]; then
	echo "[*] Skipping mSWEEP (replicate: ${replicate}, n_reads: ${n_reads}, trial: ${trial}, time_point: ${time_point})"
	exit 0
fi


echo "[*] Preparing mSWEEP input for timepoint ${time_point} (replicate: ${replicate}, n_reads=${n_reads}, trial=${trial})"
fwd_input=${output_dir}/${time_point}_pseudoaligns_1.txt
rev_input=${output_dir}/${time_point}_pseudoaligns_2.txt

echo "${fwd_input}"
if ! [ -f ${pseudoalignment_dir}/${time_point}_sim_1.output.txt ]; then echo "Forward read pseudoalignment not found."; exit 1; fi
if ! [ -f ${pseudoalignment_dir}/${time_point}_background_1.output.txt ]; then echo "Forward read pseudoalignment not found."; exit 1; fi
if ! [ -f ${pseudoalignment_dir}/${time_point}_sim_2.output.txt ]; then echo "Reverse read pseudoalignment not found."; exit 1; fi
if ! [ -f ${pseudoalignment_dir}/${time_point}_background_2.output.txt ]; then echo "Reverse read pseudoalignment not found."; exit 1; fi

cat ${pseudoalignment_dir}/${time_point}_sim_1.output.txt > $fwd_input
cat ${pseudoalignment_dir}/${time_point}_sim_2.output.txt > $rev_input
cat ${pseudoalignment_dir}/${time_point}_background_1.output.txt >> $fwd_input
cat ${pseudoalignment_dir}/${time_point}_background_2.output.txt >> $rev_input


echo "[*] Running mSWEEP"
cd ${output_dir}
start_time=$(date +%s%N)  # nanoseconds
echo "USING CLUSTERS FROM ${themisto_db_dir}"
mSWEEP \
  --themisto-1 ${fwd_input} \
  --themisto-2 ${rev_input} \
  -i ${themisto_db_dir}/clusters.txt \
  -t ${N_CORES} \
  -o ${time_point}


# ====== Record runtime
end_time=$(date +%s%N)
elapsed_time=$(( $(($end_time-$start_time)) / 1000000 ))
echo "${elapsed_time}" > $runtime_file


echo "[*] Cleaning up."
rm ${fwd_input}
rm ${rev_input}

