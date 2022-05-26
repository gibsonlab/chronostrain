#!/bin/bash
set -e
source settings.sh

export PATH=${PATH}:${GT_PRO_BIN_DIR}
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
for t_idx in 0 1 2 3 4; do
	echo "[*] Running 'GT_Pro genotype' for timepoint ${t_idx}..."
	concat_reads=${t_idx}_reads.fq.gz
	cat ${read_dir}/${t_idx}_reads_1.fq.gz ${read_dir}/${t_idx}_reads_2.fq.gz > $concat_reads
	GT_Pro genotype -f \
	-d ${GT_PRO_DB_DIR}/${GT_PRO_DB_NAME}/${GT_PRO_DB_NAME} \
	-t ${N_CORES} \
	-o ${output_dir}/${t_idx}_reads_1 \
	$concat_reads

	metagenotype="${t_idx}_reads_1.tsv"
	rm $concat_reads
	pigz -d ${metagenotype}.gz
done

#sfacts load --gtpro-metagenotype ${}
