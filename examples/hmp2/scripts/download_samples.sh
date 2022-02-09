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
	result=$(esearch -db sra -query $query | efetch -format runinfo | cut -d "," -f 1 | tail -n +2)

	if [[ $(echo "result" | wc -l) -ge 2 ]]; then
		echo "Found multiple hits for query ${query}"
		exit 1;
	fi

	echo "${result}"
}

# ================================= Main script ==================================
mkdir -p ${SAMPLES_DIR}
mkdir -p ${SRA_PREFETCH_DIR}
mkdir -p ${FASTERQ_TMP_DIR}

HMP2_CSV_PATH="${SAMPLES_DIR}/hmp2_metadata.csv"

echo "[*] Downloading HMP2 metadata file."
curl -o ${HMP2_CSV_PATH} "https://ibdmdb.org/tunnel/products/HMP2/Metadata/hmp2_metadata.csv"

# ================== Parse CSV file.
{
	# Skip header line.
	read

	# Read rest of csv file.
	while IFS=, read -r project_id external_id participant_id site_sub_coll data_type week_num date_of_receipt
	do
		if [[ "${participant_id}" =~ ^(C3022|M2072|M2079|H4008|P6012|P6035|C3015|M2069)$ ]]; then
			continue
		fi

		if [[ "${data_type}" != "metagenomics" ]]; then
			continue
		fi

		echo "[*] -=-=-=-= Downloading ${site_sub_coll}. =-=-=-=-"
		echo "[*] Querying entrez."
		query="(${project_id} OR ${external_id}) AND WGS[Strategy]"
		sra_id=$(get_sra_id "$query")
		echo "[*] SRA ID: ${sra_id}"

		# Target gzipped fastq files.
		gz_file_1="${SAMPLES_DIR}/${site_sub_coll}_1.fastq.gz"
		gz_file_2="${SAMPLES_DIR}/${site_sub_coll}_2.fastq.gz"
		if [[ -f $gz_file_1 && -f $gz_file_2 ]]; then
			echo "[*] Target files for ${sra_id} already exist."
			continue
		fi

		# Prefetch
		echo "[*] Prefetching..."
		prefetch --output-directory $SRA_PREFETCH_DIR --progress --verify yes $sra_id

		# Fasterq-dump
		echo "[*] Invoking fasterq-dump..."
		fasterq-dump \
		--progress \
		--outdir $SAMPLES_DIR \
		--skip-technical \
		--print-read-nr \
		--force \
		-t ${FASTERQ_TMP_DIR} \
		"${SRA_PREFETCH_DIR}/${sra_id}/${sra_id}.sra"

		# Resulting fq files
		fq_file_1="${SAMPLES_DIR}/${sra_id}_1.fastq"
		fq_file_2="${SAMPLES_DIR}/${sra_id}_2.fastq"

		# Compression
		echo "[*] Compressing..."
		pigz $fq_file_1 -c > $gz_file_1
		pigz $fq_file_2 -c > $gz_file_2
		wait
	done
} < ${HMP2_CSV_PATH}
