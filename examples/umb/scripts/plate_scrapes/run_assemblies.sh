#!/bin/bash
set -e
source settings.sh


# ====================== Functions ======================
handle_sample()
{
    sample_name=$1

		cd ${BASE_DIR}/scripts/plate_scrapes
		bash helpers/assemble.sh $sample_name
		bash run_blast.sh $sample_name
		cd -
}


cd $SAMPLES_DIR
for fq_file_1 in *-R1.fastq.gz; do
	# Find mate pair read file.
	regex_with_suffix="(.*)-R1.fastq.gz"
	if [[ $fq_file_1 =~ $regex_with_suffix ]]
	then
		sample_name="${BASH_REMATCH[1]}"
	else
		echo "Unexpected error; regex doesn't match."
		exit 1
	fi

	echo "[*] Handling sample ${sample_name}..."
	handle_sample $sample_name
done

