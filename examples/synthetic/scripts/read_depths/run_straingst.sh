#!/bin/bash
set -e
source settings.sh

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

# ============ script body:
trial_dir=$(get_trial_dir $n_reads $trial)
read_dir=${trial_dir}/reads
output_dir=${trial_dir}/output/straingst

echo "[*] Running inference for n_reads: ${n_reads}, trial: ${trial}, timepoint #${time_point}"

mkdir -p ${output_dir}
hdf5_path=${output_dir}/reads.hdf5

reads_1="${read_dir}/${time_point}_reads_1.fq.gz"
reads_2="${read_dir}/${time_point}_reads_2.fq.gz"

echo "[*] Kmerizing..."
straingst kmerize -k 23 -o ${hdf5_path} ${reads_1} ${reads_2}

echo "[*] Running StrainGST (minhash)."
straingst run \
-o ${output_dir}/output_mash_${time_point}.tsv \
--score 0.000001 \
${STRAINGST_DB_HDF5} \
${hdf5_path}

echo "[*] Running StrainGST (full db)."
straingst run \
-o ${output_dir}/output_fulldb_${time_point}.tsv \
--score 0.000001 \
--fulldb \
${STRAINGST_DB_HDF5} \
${hdf5_path}

echo "[*] Cleaning up."
rm ${hdf5_path}
