#!/bin/bash

# ======================================== Functions ===============================
source settings.sh

check_program 'esearch'
check_program 'efetch'
check_program 'prefetch'
check_program 'fasterq-dump'
check_program 'pigz'
check_program 'curl'


patient_subdir()
{
	target_patient="$1"
	echo "${SAMPLES_DIR}/${target_patient}"
}


get_sra_id()
{
	query="$1"
	result=$(esearch -db sra -query "$query" | efetch -format runinfo | cut -d "," -f 1 | tail -n +2)
	echo "${result}"
}


# ================== Download sample.
download_sample()
{
	participant_id=$1
	project_id=$2
	external_id=$3
	date_of_receipt=$4
	index_path=$5
	subdir=$(patient_subdir "$target_patient" | tr -d '\r')

	echo "[*] -=-=-=-= Downloading ${project_id} (participant ${participant_id}). =-=-=-=-"

	# Target gzipped fastq files.
	gz_file_1="$subdir/${project_id}_1.fastq.gz"
	gz_file_2="$subdir/${project_id}_2.fastq.gz"
	if [[ -f $gz_file_1 && -f $gz_file_2 ]]; then
		echo "[*] Target files for ${project_id} already exist."
	else
		mkdir -p $subdir

		# Entrez query to find sra_id.
		echo "[*] Querying entrez."

		query="(${project_id} OR ${external_id}) AND WGS[Strategy]"
		# https://unix.stackexchange.com/questions/342238/while-loop-execution-stops-after-one-iteration-within-bash-script
		sra_id=$(get_sra_id "$query" < /dev/null | tr -d '\r')

		if [[ $(echo "$sra_id" | wc -l) -ge 2 ]]; then
			echo "Multiple hits found for query ${query}. Using the first result only."
			sra_id=$(echo "$sra_id" | head -n 1)
		fi

		if [[ "${sra_id}" == '' ]]; then
			echo "No SRA entry found."
			return
		fi
		echo "[*] SRA ID: ${sra_id}"

		# Prefetch
		echo "[*] Prefetching..."
		prefetch --output-directory "$SRA_PREFETCH_DIR" --progress --verify yes "$sra_id"

		# Fasterq-dump
		echo "[*] Invoking fasterq-dump..."
		fasterq-dump \
		--progress \
		--outdir $subdir \
		--skip-technical \
		--print-read-nr \
		--force \
		-t ${FASTERQ_TMP_DIR} \
		"${SRA_PREFETCH_DIR}/${sra_id}/${sra_id}.sra"

		# Resulting fq files
		fq_file_1="$subdir/${sra_id}_1.fastq"
		fq_file_2="$subdir/${sra_id}_2.fastq"

		# Compression
		echo "[*] Compressing..."
		pigz "$fq_file_1" -c > "$gz_file_1"
		pigz "$fq_file_2" -c > "$gz_file_2"
		rm "$fq_file_1"
		rm "$fq_file_2"
	fi

	echo "${project_id},${date_of_receipt}" >> "$index_path"
}


# ================== Parse CSV file.

target_patient=$1

# Prepare index file.
subdir=$(patient_subdir "$target_patient" | tr -d '\r')
mkdir -p "${subdir}"

index_path="${subdir}/index.csv"
> ${index_path}  # clear contents

# Sed: Skip the header line of HMP2 metadata CSV file.
# Awk: parse appropriate fields from CSV.
sed 1d ${HMP2_CSV_PATH} | \
awk -F, -v target_patient="$target_patient" '{
	project_id=$1
	external_id=$2
	participant_id=$3
	data_type=$5
	date_of_receipt=$7
	if (participant_id == target_patient && data_type == "metagenomics") {
		printf "%s,%s,%s,%s\n", project_id, external_id, participant_id, date_of_receipt
	}
}' | \
while IFS=, read -r project_id external_id participant_id date_of_receipt; do
	download_sample "$participant_id" "$project_id" "$external_id" "$date_of_receipt" "$index_path"
done
