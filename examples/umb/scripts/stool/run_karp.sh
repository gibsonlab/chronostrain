#!/bin/bash
set -e

source settings.sh
check_program pigz
check_program karp


do_inference() {
	sra_id=$1
	sample_id=$2
	patient=$3

	out_dir=${KARP_OUTPUT_DIR}/${patient}
	out_prefix="${sample_id}"
	output_file="${out_prefix}.freqs"
	read_counts="out_prefix.reads"

	file1="${SAMPLES_DIR}/kneaddata/${sra_id}/${sra_id}_paired_1.fastq.gz"
  file2="${SAMPLES_DIR}/kneaddata/${sra_id}/${sra_id}_unmatched_1.fastq.gz"
  file3="${SAMPLES_DIR}/kneaddata/${sra_id}/${sra_id}_paired_2.fastq.gz"
  file4="${SAMPLES_DIR}/kneaddata/${sra_id}/${sra_id}_unmatched_2.fastq.gz"

	mkdir -p $out_dir
	cd $out_dir

  if [ -f $output_file ]; then
  	echo "[*] Karp result on ${sample_id} (patient ${patient}) already found."
  else
  	echo "[*] Counting reads."
		n_reads_1=$(($(pigz -dc $file1 | wc -l | awk '{print $1}') / 4))
		n_reads_2=$(($(pigz -dc $file2 | wc -l | awk '{print $1}') / 4))
		n_reads_3=$(($(pigz -dc $file3 | wc -l | awk '{print $1}') / 4))
		n_reads_4=$(($(pigz -dc $file4 | wc -l | awk '{print $1}') / 4))
		> $read_counts
		echo -e "${Paired_1}\t${n_reads_1}" >> $read_counts
		echo -e "${Unmatched_1}\t${n_reads_2}" >> $read_counts
		echo -e "${Paired_2}\t${n_reads_3}" >> $read_counts
		echo -e "${Unmatched_2}\t${n_reads_4}" >> $read_counts

		echo "[*] Running Karp on ${sample_id} (patient ${patient})."
		karp \
			-c quantify \
			-r ${KARP_REFS} \
			-i ${KARP_IDX} \
			-t ${KARP_TAX} \
			-f $file1,$file2,$file3,$file4 \
			--phred 33 \
			--threads 4 \
			-o $out_prefix
	fi

	cd -
}


# ================== Parse CSV file.
{
	# Skip header line.
	read

	# Read rest of csv file.
	while IFS=, read -r sra_id umb_id sample_name date days experiment_type model library_strategy exp_group
	do
		if [ "$experiment_type" = "stool" ] && [ "$umb_id" = "UMB18" ]; then
			do_inference $sra_id $sample_name $umb_id
		fi
	done
} < ${SRA_CSV_PATH}
