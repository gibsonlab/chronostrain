#!/bin/bash
set -e
source settings.sh

# ======================================== Functions ===============================
# REQUIRES: sratools (prefetch + fasterq-dump), pigz
check_program 'esearch'
check_program 'efetch'
check_program 'prefetch'
check_program 'fasterq-dump'
check_program 'pigz'
check_program 'curl'

get_sra_id()
{
	query="$1"
	result=$(esearch -db sra -query "$query" | efetch -format runinfo | cut -d "," -f 1 | tail -n +2)
	echo "${result}"
}

# ================================= Main script ==================================
mkdir -p ${SAMPLES_DIR}
mkdir -p ${SRA_PREFETCH_DIR}
mkdir -p ${FASTERQ_TMP_DIR}

HMP2_CSV_PATH="${SAMPLES_DIR}/hmp2_metadata.csv"
FILE_INDEX_PATH="${SAMPLES_DIR}/index.csv"

echo "[*] Downloading HMP2 metadata file."
curl -o ${HMP2_CSV_PATH} "https://ibdmdb.org/tunnel/products/HMP2/Metadata/hmp2_metadata.csv"

# Clear the contents of the file.
> ${FILE_INDEX_PATH}

# ================== Parse CSV file.
download_patient_samples()
{
	target_patient=$1
	{
		# Skip header line.
		read

		# Read rest of csv file.
		while IFS=, read -r project_id external_id participant_id site_sub_coll data_type week_num date_of_receipt
		do
			if [[ "${participant_id}" != "${target_patient}" ]]; then
				continue
			fi

			if [[ "${data_type}" != "metagenomics" ]]; then
				continue
			fi

			echo "[*] -=-=-=-= Downloading ${project_id}. =-=-=-=-"
			echo "[*] Querying entrez."

			query="(${project_id} OR ${external_id}) AND WGS[Strategy]"
			sra_id=$(get_sra_id "$query")
			if [[ $(echo "$sra_id" | wc -l) -ge 2 ]]; then
				echo "Multiple hits found for query ${query}. Using the first result only."
				sra_id=$(echo "$sra_id" | head -n 1)
			fi

			echo "[*] SRA ID: ${sra_id}"

			# Target gzipped fastq files.
			subdir="${SAMPLES_DIR}/${participant_id}"
			gz_file_1="$subdir/${project_id}_1.fastq.gz"
			gz_file_2="$subdir/${project_id}_2.fastq.gz"
			if [[ -f $gz_file_1 && -f $gz_file_2 ]]; then
				echo "[*] Target files for ${sra_id} already exist."
			else
				mkdir -p $subdir

				# Prefetch
				echo "[*] Prefetching..."
				prefetch --output-directory $SRA_PREFETCH_DIR --progress --verify yes $sra_id

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
				pigz $fq_file_1 -c > $gz_file_1
				pigz $fq_file_2 -c > $gz_file_2
				wait
			fi

			echo "${participant_id},${project_id},${sra_id}" >> ${FILE_INDEX_PATH}
		done
	} < ${HMP2_CSV_PATH}
}

# nonIBD
download_patient_samples "C3022"
download_patient_samples "M2072"
download_patient_samples "M2079"
download_patient_samples "H4008"

# UC
download_patient_samples "P6012"
download_patient_samples "P6035"
download_patient_samples "C3015"
download_patient_samples "M2069"
