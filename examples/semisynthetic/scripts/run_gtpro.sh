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

echo "[*] Running GTPro metagenotyping for n_reads: ${n_reads}, trial: ${trial}"

mkdir -p ${output_dir}
cd ${output_dir}


mg_prefix="mg_all"
metagenotype_all="${mg_prefix}.tsv"


# Run metagenotyping
echo "[*] Preparing GT_Pro inputs..."
for t_idx in 0 1 2 3 4; do
	concat_reads=${t_idx}_reads.fq
	> $concat_reads

	pigz -dck ${read_dir}/${t_idx}_reads_1.fq.gz >> $concat_reads
	pigz -dck ${read_dir}/${t_idx}_reads_2.fq.gz >> $concat_reads

#	cat ${read_dir}/${t_idx}_NZ_CP069709.1_1.fq >> $concat_reads
#	cat ${read_dir}/${t_idx}_NZ_CP069709.1_2.fq >> $concat_reads
#
#	cat ${read_dir}/${t_idx}_NZ_CP076645.1_1.fq >> $concat_reads
#	cat ${read_dir}/${t_idx}_NZ_CP076645.1_2.fq >> $concat_reads
#
#	cat ${read_dir}/${t_idx}_NZ_CP026399.1_1.fq >> $concat_reads
#	cat ${read_dir}/${t_idx}_NZ_CP026399.1_2.fq >> $concat_reads
#
#	cat ${read_dir}/${t_idx}_NZ_LR134247.1_1.fq >> $concat_reads
#	cat ${read_dir}/${t_idx}_NZ_LR134247.1_2.fq >> $concat_reads
done


echo "[*] Running 'GT_Pro genotype'..."
start_time=$(date +%s%N)  # nanoseconds

GT_Pro genotype -f \
-d ${GT_PRO_DB_DIR}/${GT_PRO_DB_NAME}/${GT_PRO_DB_NAME} \
-t ${N_CORES} \
-o ${output_dir}/%{n}_reads \
0_reads.fq 1_reads.fq 2_reads.fq 3_reads.fq 4_reads.fq


echo "[*] Parsing GT_Pro outputs..."
> $metagenotype_all  # Clear file incase it exists.

for t_idx in 0 1 2 3 4; do
	metagenotype="${t_idx}_reads.tsv"
	GT_Pro parse --dict ${GT_PRO_DB_DIR}/${GT_PRO_DB_NAME}/${GT_PRO_DB_NAME}.snp_dict.noheader.tsv --in $metagenotype \
	| awk -v t="${t_idx}" '{if (NR!=1) {print t "\t" $0;}}' >> $metagenotype_all
done

# ====== Record runtime
end_time=$(date +%s%N)
elapsed_time=$(( $(($end_time-$start_time)) / 1000000 ))
runtime_file=${trial_dir}/output/gtpro_runtime.txt
echo "${elapsed_time}" > $runtime_file

echo "[*] Cleaning up..."
rm *.fq