#!/bin/bash
set -e
source settings.sh


# ====================== Functions ======================
add_sample()
{
    participant=$1
    source_sample=$2
    culture_name=$3
    fq_1=$4
    fq_2=$5

    num_lines_1=$(pigz -dc $fq_1 | wc -l | awk '{print $1}')
    num_reads_1=$((${num_lines_1} / 4))
    num_lines_2=$(pigz -dc $fq_2 | wc -l | awk '{print $1}')
    num_reads_2=$((${num_lines_2} / 4))

    reads_file=${OUTPUT_DIR}/${participant}/${source_sample}/reads.csv
    mkdir -p ${OUTPUT_DIR}/${participant}/${source_sample}
    echo "1000,${culture_name},${num_reads_1},${fq_1},paired_1,fastq" >> $reads_file
    echo "1000,${culture_name},${num_reads_2},${fq_2},paired_2,fastq" >> $reads_file
}


shopt -s nullglob
for f in ${OUTPUT_DIR}/UMB*/*/reads.csv; do rm $f; done

while IFS=$'\t' read participant source culture
do
	if [ "${participant}" == "Participant" ]; then continue; fi  # skip header
	echo "Found culture seq id: ${culture}"

	fq_1=${SAMPLES_DIR}/Esch_coli_${culture}-R1.fastq.gz
	fq_2=${SAMPLES_DIR}/Esch_coli_${culture}-R2.fastq.gz
	if [ ! -f ${fq_1} ]; then echo "Couldn't find fastq file ${fq_1}"; continue; fi
	if [ ! -f ${fq_2} ]; then echo "Couldn't find fastq file ${fq_2}"; continue; fi
	add_sample $participant $source $culture $fq_1 $fq_2
done < ${BASE_DIR}/files/plate_samples.tsv
