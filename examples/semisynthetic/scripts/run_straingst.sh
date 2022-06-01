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

# ============ script body:
trial_dir=$(get_trial_dir $n_reads $trial)
read_dir=${trial_dir}/reads
output_dir=${trial_dir}/output/straingst

mkdir -p ${output_dir}


echo "[*] Running inference for n_reads: ${n_reads}, trial: ${trial}, timepoint #${time_point}"
start_time=$(date +%s%N)  # nanoseconds


read_kmers=${output_dir}/reads.hdf5
reads_1="${read_dir}/${time_point}_reads_1.fq.gz"
reads_2="${read_dir}/${time_point}_reads_2.fq.gz"

echo "[*] Kmerizing..."
straingst kmerize -k 23 -o ${read_kmers} ${reads_1} ${reads_2}

echo "[*] Running StrainGST."
mkdir -p ${output_dir}/${mode}
straingst run \
-o ${output_dir}/${mode}/output_mash_${time_point}.tsv \
${straingst_db} \
${read_kmers}

# ====== Record runtime
end_time=$(date +%s%N)
elapsed_time=$(( $(($end_time-$start_time)) / 1000000 ))
runtime_file=${trial_dir}/straingst_runtime.txt
echo "${elapsed_time}" > $runtime_file


echo "[*] Cleaning up."
rm ${read_kmers}
