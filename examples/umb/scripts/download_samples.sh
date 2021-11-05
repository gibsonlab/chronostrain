#!/bin/bash
set -e
source settings.sh

# REQUIRES: wget, sratools, kneaddata, gzip
check_program()
{
	command -v ${1} >/dev/null 2>&1 || {
		echo >&2 "I require ${1} but it's not installed.  Aborting.";
		exit 1;
	}
}
check_program 'wget'
check_program 'gzip'
check_program 'kneaddata'
check_program 'prefetch'
check_program 'fasterq-dump'

# Gzips the input fastq file, and appends the fastq-timepoint pair as an entry.
gzip_and_append_fastq()
{
	fq_path=$1
	time=$2
	num_lines=$(wc -l $fq_path | awk '{print $1}')
	num_reads=$((${num_lines} / 4))
	if [[ $num_reads > 0 ]]; then
  	echo "[*] Compressing: ${fq_path}"
		gzip $fq_path --force
		gz_file="${fq_path}.gz"
		echo "\"${time}\",\"${num_reads}\",\"${gz_file}\"" >> $INPUT_INDEX_PATH
  else
  	echo "Skipping ${fq_path}."
  fi
}

# Clear index file.
mkdir -p ${READS_DIR}
touch $INPUT_INDEX_PATH
> $INPUT_INDEX_PATH

SRA_CSV_PATH="${SAMPLES_DIR}/SraRunInfo.csv"
query="${BIOPROJECT}+AND+UMB24+AND+stool"

mkdir -p ${SAMPLES_DIR}
mkdir -p ${SRA_PREFETCH_DIR}

# Download seq repository.
wget \
"http://trace.ncbi.nlm.nih.gov/Traces/sra/sra.cgi?save=efetch&rettype=runinfo&db=sra&term=${query}" \
-O "${SRA_CSV_PATH}"

# ================== Parse CSV file.
sra_id_col='Run'
lib_name_col='LibraryName'

loc_col_a=$(head -1 ${SRA_CSV_PATH} | tr ',' '\n' | nl |grep -w "$sra_id_col" | tr -d " " | awk -F " " '{print $1}')
loc_col_b=$(head -1 ${SRA_CSV_PATH} | tr ',' '\n' | nl |grep -w "$lib_name_col" | tr -d " " | awk -F " " '{print $1}')

while IFS="," read -r sra_id lib_name
do
	if [ -z "${sra_id}" ] || [ -z "${lib_name}" ]; then
    continue
  fi

  if [[ $lib_name =~ [a-zA-Z]+$ ]]; then
  	echo "Skipping ${lib_name}."
  	continue
  fi

	echo "-=-=-=-=-=-=-= Handling ${sra_id} (${lib_name}). =-=-=-=-=-=-=-"

  # Extract token from UMB(patient)_(token).xyz
  suffix=${lib_name##*_}
  month=${suffix%.*}
  time=$((10#${month} * 30))  # 30 days per month is a crude estimate.

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
	"${SRA_PREFETCH_DIR}/${sra_id}/${sra_id}.sra"

	# Obtained fastq files.
	fq_file_1="${SAMPLES_DIR}/${sra_id}_1.fastq"
	fq_file_2="${SAMPLES_DIR}/${sra_id}_2.fastq"

	# Preprocess
	echo "[*] Invoking kneaddata..."
	echo "KNEADDATA DIR: ${KNEADDATA_DB_DIR}"
	echo "kneaddata --input1 $fq_file_1 --input2 $fq_file_2 -db ${KNEADDATA_DB_DIR} --output ${SAMPLES_DIR}/kneaddata_output --trimmomatic $TRIMMOMATIC_DIR --trimmomatic-options SLIDINGWINDOW:100:0 MINLEN:35 --sequencer-source NexteraPE --bypass-trf"
	kneaddata \
	--input $fq_file_1 \
	--input $fq_file_2 \
	-db ${KNEADDATA_DB_DIR} \
	--output ${SAMPLES_DIR}/kneaddata_output \
	--trimmomatic $TRIMMOMATIC_DIR \
	--trimmomatic-options "SLIDINGWINDOW:100:0 MINLEN:35" \
	--sequencer-source NexteraPE \
	--bypass-trf

	gzip_and_append_fastq "${SAMPLES_DIR}/kneaddata_output/${sra_id}_1_kneaddata_paired_1.fastq"
	gzip_and_append_fastq "${SAMPLES_DIR}/kneaddata_output/${sra_id}_1_kneaddata_unmatched_1.fastq"
done < <(cut -d "," -f${loc_col_a},${loc_col_b} ${SRA_CSV_PATH} | tail -n +2)
