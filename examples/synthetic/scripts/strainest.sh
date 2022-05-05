#!/bin/bash
set -e
source settings.sh

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
output_dir=${trial_dir}/output/strainest

mkdir -p ${output_dir}
cd ${output_dir}

# Concatenate reads.
reads_1="reads_${time_point}.1.fq"
reads_2="reads_${time_point}.2.fq"
reads_1_gz="${reads_1}.gz"
reads_2_gz="${reads_2}.gz"
echo "[*] Concatenating reads..."
pigz -dck ${read_dir}/${time_point}_CP009273.1_Original_1.fq.gz > ${reads_1}
pigz -dck ${read_dir}/${time_point}_CP009273.1_Original_2.fq.gz > ${reads_2}
pigz -dck ${read_dir}/${time_point}_CP009273.1_Substitution_1.fq.gz > ${reads_1}
pigz -dck ${read_dir}/${time_point}_CP009273.1_Substitution_2.fq.gz > ${reads_2}
pigz ${reads_1}
pigz ${reads_2}

# Compile bowtie2 database
echo "TODO"

# Perform bowtie2 alignment
sam_file="reads_${time_point}.sam"
bam_file="reads_${time_point}.bam"
sorted_bam_file="reads_${time_point}.sorted.bam"

cd /mnt/d/chronostrain/synthetic/database/straingst
bowtie2 \
--very-fast --no-unal \
-x ${BOWTIE2_INDEX_BASENAME} \
-1 ${reads_1_gz} \
-2 ${reads_2_gz} \
-S ${output_dir}/${sam_file}
cd ${output_dir}

# Invoke samtools
samtools view -b ${sam_file} > ${bam_file}
samtools sort ${bam_file} -o ${sorted_bam_file}
samtools index ${sorted_bam_file}

# Run StrainEst
strainest est

# Clean up
rm ${reads_1_gz}
rm ${reads_2_gz}
rm ${sam_file}
rm ${bam_file}
