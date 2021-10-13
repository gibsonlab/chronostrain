#!/bin/bash
set -e

# Requires sratools!

source settings.sh

# Clear index file.
mkdir -p ${READS_DIR}
touch $INPUT_INDEX_PATH
> $INPUT_INDEX_PATH

SRA_CSV_PATH="${SAMPLES_DIR}/SraRunInfo.csv"
query="${BIOPROJECT}+AND+UMB24+AND+stool"

mkdir -p ${SAMPLES_DIR}
mkdir -p ${SRA_PREFETCH_DIR}

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

	# Prefetch
	echo "[*] Prefetching ${sra_id}..."
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

	# Gzip
	fq_file_1="${SAMPLES_DIR}/${sra_id}_1.fastq"
	fq_file_2="${SAMPLES_DIR}/${sra_id}_2.fastq"
	echo "[*] Compressing..."
	gzip $fq_file_1 --force

	# ===== TEMPORARY: we aren't properly using paired-end information.
#	rm $fq_file_2

	# Create index
	time=1
	n_reads=100
	gz_file="${fq_file_1}.gz"
	echo "\"${time}\",\"${n_reads}\",\"${gz_file}\"" >> $INPUT_INDEX_PATH
done < <(cut -d "," -f${loc_col_a},${loc_col_b} ${SRA_CSV_PATH} | tail -n +2)
