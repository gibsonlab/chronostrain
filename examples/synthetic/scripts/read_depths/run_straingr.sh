#!/bin/bash
set -e
source settings.sh

# ============ Requires arguments:
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

# ============ script body:
trial_dir=$(get_trial_dir $n_reads $trial)
read_dir=${trial_dir}/reads
straingst_output_dir=${trial_dir}/output/straingst
straingr_output_dir=${trial_dir}/output/straingr

echo "[*] Running StrainGR for n_reads: ${n_reads}, trial: ${trial}, timepoint #${time_point}"

mkdir -p ${straingr_output_dir}
cd ${straingr_output_dir}


echo "[*] Running StrainGR 'prepare-ref'."
straingr prepare-ref \
--output refs_concat.fasta \
-S ${STRAINGST_DB_DIR}/similarities.tsv \
-t 1.0 \
-s \
${straingst_output_dir}/output_fulldb_0.tsv \
${straingst_output_dir}/output_fulldb_1.tsv \
${straingst_output_dir}/output_fulldb_2.tsv \
${straingst_output_dir}/output_fulldb_3.tsv \
${straingst_output_dir}/output_fulldb_4.tsv \
-r "CP009273.1_Original" "CP009273.1_Substitution" \
-p "${STRAINGST_DB_DIR}/{ref}.fasta"


for time_point in 0 1 2 3 4
do
	echo "[*] Handling timepoint ${time_point}..."
	reads_1="${read_dir}/${time_point}_reads_1.fq.gz"
	reads_2="${read_dir}/${time_point}_reads_2.fq.gz"

	echo "[*] Aligning..."
	bam_file="sample_${time_point}.bam"
	summary_tsv="sample_${time_point}.tsv"
	hdf5_file="sample_${time_point}.hdf5"

	bwa index refs_concat.fasta
	bwa mem -I 300 -t 4 refs_concat.fasta ${reads_1} ${reads_2} \
	| samtools sort -@ 2 -O BAM -o ${bam_file} -
	samtools index ${bam_file}

	echo "[*] Running StrainGR 'call'."
	straingr call refs_concat.fasta ${bam_file} --summary ${summary_tsv} --tracks all --hdf5-out ${hdf5_file}
done
