#!/bin/bash
set -e
source settings.sh

# ============ Requires arguments:
trial_dir=$1

if [ -z "$trial_dir" ]
then
	echo "var \"trial_dir\" is empty"
	exit 1
fi

# ============ script body:
read_dir=${trial_dir}/reads
log_dir=${trial_dir}/logs
trim_dir=${read_dir}/trimmomatic
mkdir -p ${trim_dir}


append_fastq()
{
	fq_path=$1
	time_value=$2
	csv_file=$3
	read_type=$4

	num_lines=$(wc -l $fq_path | awk '{print $1}')
	num_reads=$((${num_lines} / 4))

	pigz ${fq_path}
	echo "${time_value},${num_reads},${fq_path}.gz,${read_type},fastq" >> $csv_file
}


# Read in timepoints.
time_points=()
while IFS="," read time_point; do
	array+=($time_point)
done < ${GROUND_TRUTH}

index_csv="input_files_trimmed.csv"
> $index_csv

for time_point in 0 1 2 3 4; do
	echo "[*] Running trimmomatic on timepoint #${time_point}..."
	t_value="${time_points[time_point]}"

	reads_1="${read_dir}/${time_point}_reads_1.fq.gz"
	reads_2="${read_dir}/${time_point}_reads_2.fq.gz"
	trimmed_paired_1="${trim_dir}/${time_point}_reads_1.paired.fq"
	trimmed_unpaired_1="${trim_dir}/${time_point}_reads_1.unpaired.fq"
	trimmed_paired_2="${trim_dir}/${time_point}_reads_2.paired.fq"
	trimmed_unpaired_2="${trim_dir}/${time_point}_reads_2.unpaired.fq"

	trimmomatic \
	PE \
	-threads 4 \
	-phred33 \
	-trimlog ${log_dir}/trimmomatic_{time_point}.log \
	-quiet \
	${reads_1} ${reads_2} \
	${trimmed_paired_1} ${trimmed_unpaired_1} \
	${trimmed_paired_2} ${trimmed_unpaired_2} \
	LEADING:10 TRAILING:10 MINLEN:35

	append_fastq ${trimmed_paired_1} ${t_value} ${index_csv} "paired_1"
	append_fastq ${trimmed_unpaired_1} ${t_value} ${index_csv} "paired_1"
	append_fastq ${trimmed_paired_2} ${t_value} ${index_csv} "paired_2"
	append_fastq ${trimmed_unpaired_2} ${t_value} ${index_csv} "paired_2"
done
