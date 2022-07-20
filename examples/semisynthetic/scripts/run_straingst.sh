#!/bin/bash
set -e
source settings.sh

# ============ Requires arguments:
n_reads=$1
trial=$2
time_point=$3
mode=$4

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

if [[ "$mode" == "chromosome" ]]
then
	straingst_db=${STRAINGST_CHROMOSOME_DB_HDF5}
elif [[ "$mode" == "marker" ]]
then
	straingst_db=${STRAINGST_MARKER_DB_HDF5}
else
	echo "var \"mode\" must be either \"chromosome\" or \"marker\"."
	exit 1
fi

trial_dir=$(get_trial_dir $n_reads $trial)
read_dir=${trial_dir}/reads
output_dir=${trial_dir}/output/straingst

mkdir -p ${output_dir}


# ========== Run
echo "[*] Running inference for n_reads: ${n_reads}, trial: ${trial}, timepoint #${time_point}"
start_time=$(date +%s%N)  # nanoseconds
read_kmers=${output_dir}/reads.hdf5


echo "[*] Kmerizing..."
straingst kmerize \
-k 23 \
-o ${read_kmers} \
${read_dir}/${time_point}_sim_1.fq \
${BACKGROUND_FASTQ_DIR}/${time_point}_background_1.fq \
${read_dir}/${time_point}_sim_2.fq \
${BACKGROUND_FASTQ_DIR}/${time_point}_background_2.fq


echo "[*] Running StrainGST."
mkdir -p ${output_dir}/${mode}
straingst run \
-o ${output_dir}/${mode}/output_mash_${time_point}.tsv \
${straingst_db} \
${read_kmers}

# ====== Record runtime
end_time=$(date +%s%N)
elapsed_time=$(( $(($end_time-$start_time)) / 1000000 ))
runtime_file=${trial_dir}/output/straingst_runtime.${time_point}.${mode}.txt
echo "${elapsed_time}" > $runtime_file


echo "[*] Cleaning up."
rm ${read_kmers}
