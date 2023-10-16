#!/bin/bash
set -e
source settings.sh


# ====================== Functions ======================
add_sample()
{
    sample_name=$1
    fq_1=$2
    fq_2=$3

    num_lines_1=$(pigz -dc $fq_1 | wc -l | awk '{print $1}')
    num_reads_1=$((${num_lines_1} / 4))
    num_lines_2=$(pigz -dc $fq_2 | wc -l | awk '{print $1}')
    num_reads_2=$((${num_lines_2} / 4))

    reads_file=${OUTPUT_DIR}/${sample_name}/reads.csv
    mkdir -p ${OUTPUT_DIR}/${sample_name}
    echo "1000,${sample_name},${num_reads_1},${SAMPLES_DIR}/${fq_1},paired_1,fastq" > $reads_file  # Clear previous content, hence > and not >>
    echo "1000,${sample_name},${num_reads_2},${SAMPLES_DIR}/${fq_2},paired_2,fastq" >> $reads_file  # Append
}


cd $SAMPLES_DIR
for fq_file_1 in *-R1.fastq.gz; do
	# Find mate pair read file.
	regex_with_suffix="(.*)-R1.fastq.gz"
	if [[ $fq_file_1 =~ $regex_with_suffix ]]
	then
		sample_name="${BASH_REMATCH[1]}"
		fq_file_2="${sample_name}-R2.fastq.gz"
	else
		echo "Unexpected error; regex doesn't match."
		exit 1
	fi

	echo "[*] Found: $fq_file_1; mate pair=$fq_file_2"
	add_sample $sample_name $fq_file_1 $fq_file_2
done

