#!/bin/bash
set -e
source settings.sh
source msweep/settings.sh

require_program mSWEEP

# ============ Requires arguments:
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

themisto_db_dir="${DATA_DIR}/databases/themisto"
trial_dir=$(get_trial_dir "${mutation_ratio}" "$replicate" "$n_reads" "$trial")
output_dir=${trial_dir}/output/msweep
pseudoalignment_dir=${trial_dir}/output/themisto
runtime_file=${output_dir}/runtime.${time_point}.txt
mkdir -p $output_dir


if [ -f $runtime_file ]; then
	echo "[*] Skipping mSWEEP (mut_ratio: ${mutation_ratio} | replicate: ${replicate} |  n_reads: ${n_reads} | trial: ${trial} | timepoint ${time_point})"
	exit 0
fi


echo "[*] Preparing mSWEEP input (mut_ratio: ${mutation_ratio} | replicate: ${replicate} |  n_reads: ${n_reads} | trial: ${trial} | timepoint ${time_point})"
fwd_input=${output_dir}/${time_point}_pseudoaligns_1.txt
rev_input=${output_dir}/${time_point}_pseudoaligns_2.txt

bg1=/mnt/e/semisynthetic_data/mutratio_1.0/replicate_1/reads_10000/trial_1/output/themisto/${time_point}_background_1.output.txt
bg2=/mnt/e/semisynthetic_data/mutratio_1.0/replicate_1/reads_10000/trial_1/output/themisto/${time_point}_background_2.output.txt
if ! [ -f ${pseudoalignment_dir}/${time_point}_sim_1.output.txt ]; then echo "Forward read pseudoalignment not found."; exit 1; fi
if ! [ -f ${bg1} ]; then echo "Forward read pseudoalignment not found."; exit 1; fi
if ! [ -f ${pseudoalignment_dir}/${time_point}_sim_2.output.txt ]; then echo "Reverse read pseudoalignment not found."; exit 1; fi
if ! [ -f ${bg2} ]; then echo "Reverse read pseudoalignment not found."; exit 1; fi

cat ${pseudoalignment_dir}/${time_point}_sim_1.output.txt > $fwd_input
cat ${pseudoalignment_dir}/${time_point}_sim_2.output.txt > $rev_input

# re-use previous pseudoalignments for the background reads.
cat ${bg1} >> $fwd_input
cat ${bg2} >> $rev_input


echo "[*] Running mSWEEP"
cd ${output_dir}
start_time=$(date +%s%N)  # nanoseconds
echo "USING CLUSTERS FROM ${themisto_db_dir}"
mSWEEP \
  --themisto-1 ${fwd_input} \
  --themisto-2 ${rev_input} \
  -i ${themisto_db_dir}/clusters.txt \
  -t ${N_CORES} \
  -o ${time_point} \
  --verbose


# ====== Record runtime
end_time=$(date +%s%N)
elapsed_time=$(( $(($end_time-$start_time)) / 1000000 ))
echo "${elapsed_time}" > $runtime_file


echo "[*] Cleaning up."
rm ${fwd_input}
rm ${rev_input}

