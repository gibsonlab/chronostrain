#!/bin/bash
set -e
source settings.sh

require_program mSWEEP

# ============ Requires arguments:
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

if [ -z "$time_point" ]
then
	echo "var \"time_point\" is empty"
	exit 1
fi

trial_dir=$(get_trial_dir $n_reads $trial)
read_dir=${trial_dir}/reads
output_dir=${trial_dir}/output/msweep
pseudoalignment_dir=${trial_dir}/output/themisto
runtime_file=${trial_dir}/output/msweep_runtime.${time_point}.txt
mkdir -p $output_dir


if [ -f $runtime_file ]; then
	echo "[*] Skipping mSWEEP (n_reads: ${n_reads}, trial: ${trial}, time_point: ${time_point})"
	exit 0
fi


echo "[*] Preparing mSWEEP input for timepoint ${time_point} (n_reads=${n_reads}, trial=${trial})"
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
mSWEEP \
  --themisto-1 ${fwd_input} \
  --themisto-2 ${rev_input} \
  -i ${THEMISTO_DB_DIR}/clusters.txt \
  -t ${N_CORES} \
  -o ${time_point}


# ====== Record runtime
end_time=$(date +%s%N)
elapsed_time=$(( $(($end_time-$start_time)) / 1000000 ))
echo "${elapsed_time}" > $runtime_file


echo "[*] Cleaning up."
rm ${fwd_input}
rm ${rev_input}

