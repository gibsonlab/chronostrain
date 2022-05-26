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
echo "[*] Preparing GT_Pro inputs..."
for t_idx in 0 1 2 3 4; do
	concat_reads=${t_idx}_reads.fq
	> $concat_reads
	pigz -dck ${read_dir}/${t_idx}_reads_1.fq.gz >> $concat_reads
	pigz -dck ${read_dir}/${t_idx}_reads_2.fq.gz >> $concat_reads
done


echo "[*] Running 'GT_Pro genotype' for timepoint ${t_idx}..."
GT_Pro genotype -f \
-d ${GT_PRO_DB_DIR}/${GT_PRO_DB_NAME}/${GT_PRO_DB_NAME} \
-t ${N_CORES} \
-o ${output_dir}/%{n}_reads.tsv \
0_reads.fq 1_reads.fq 2_reads.fq 3_reads.fq 4_reads.fq


echo "[*] Parsing GT_Pro outputs..."
mg_prefix="mg_all"
metagenotype_all="${mg_prefix}.tsv"
> $metagenotype_all  # Clear file incase it exists.

for t_idx in 0 1 2 3 4; do
	metagenotype="${t_idx}_reads.tsv"
	pigz -d ${metagenotype}.gz
	GT_Pro parse --dict ${GT_PRO_DB_DIR}/${GT_PRO_DB_NAME}/${GT_PRO_DB_NAME}.snp_dict.noheader.tsv --in $metagenotype \
	| awk -v t="${t_idx}" '{ print t "\t" $0; }' >> $metagenotype_all
done

sfacts load --gtpro-metagenotype ${metagenotype_all} $mg_prefix
sfacts fit \
--device cuda \
--precision 32 \
--num_strains 4 \
--random-seed 0 \
--optimizer-learning-rate 0.05 \
--min-optimizer-learning-rate 1e-06 \
${mg_prefix}.mgen.nc ${mg_prefix}.world.nc

echo "[*] Cleaning up..."
rm *.fq
