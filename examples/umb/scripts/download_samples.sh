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

	# Gzip
	fq_file_1="${SAMPLES_DIR}/${sra_id}_1.fastq"
	fq_file_2="${SAMPLES_DIR}/${sra_id}_2.fastq"
	num_lines=$(wc -l $fq_file_1 | awk '{print $1}')
	num_reads=$((${num_lines} / 4))

	echo "[*] Compressing..."
	gzip $fq_file_1 --force

	# ===== TEMPORARY: we aren't properly using paired-end information.
	echo "[*] Cleaning up reverse reads."
	rm $fq_file_2

	# Create index
	gz_file="${fq_file_1}.gz"
	echo "\"${time}\",\"${num_reads}\",\"${gz_file}\"" >> $INPUT_INDEX_PATH
done < <(cut -d "," -f${loc_col_a},${loc_col_b} ${SRA_CSV_PATH} | tail -n +2)
