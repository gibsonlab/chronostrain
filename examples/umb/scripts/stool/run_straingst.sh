#!/bin/bash
set -e

source settings.sh


HDF5_DIR=${STRAINGE_OUTPUT_DIR}/hdf5_files

do_inference() {
	sra_id=$1
	sample_id=$2
	patient=$3

	output_file="${STRAINGE_OUTPUT_DIR}/${patient}/${sample_id}.tsv"
	file1="/mnt/e/umb_data/samples_stool/kneaddata/${sra_id}/${sra_id}_paired_1.fastq.gz"
  file2="/mnt/e/umb_data/samples_stool/kneaddata/${sra_id}/${sra_id}_unmatched_1.fastq.gz"
  file3="/mnt/e/umb_data/samples_stool/kneaddata/${sra_id}/${sra_id}_paired_2.fastq.gz"
  file4="/mnt/e/umb_data/samples_stool/kneaddata/${sra_id}/${sra_id}_unmatched_2.fastq.gz"

  if [ -f $output_file ]; then
  	echo "[*] StrainGST result on ${sample_id} already found."
  else
		echo "[*] Running StrainGST on ${sample_id}."

		mkdir -p ${HDF5_DIR}
		straingst kmerize -k 23 -o ${HDF5_DIR}/${sample_id}.hdf5 $file1 $file2 $file3 $file4

		mkdir -p ${STRAINGE_OUTPUT_DIR}/${patient}
		straingst run -o $output_file ${STRAINGE_DB} ${HDF5_DIR}/${sample_id}.hdf5

		echo "[*] Cleaning up."
		rm ${HDF5_DIR}/${sample_id}.hdf5
	fi
}


# ================== Parse CSV file.
{
	# Skip header line.
	read

	# Read rest of csv file.
	while IFS=, read -r sra_id umb_id sample_name date days experiment_type model library_strategy exp_group
	do
		do_inference $sra_id $sample_name $umb_id
	done
} < ${SRA_CSV_PATH}
