#!/bin/bash
set -e
source settings.sh

seed=0
check_program 'art_illumina'
check_program 'prefetch'
check_program 'fasterq-dump'
check_program 'seqtk'

# First, download the background samples.
mkdir -p ${BACKGROUND_FASTQ_DIR}


# ============ Function definitions
download_sra()
{
	sra_id=$1
	gz1=$2
	gz2=$3

	if [[ -f $gz1 && -f $gz2 ]]; then
		echo "[*] Target files for ${sra_id} already exist."
	else
		sra_fq1="${sra_id}_1.fastq"
		sra_fq2="${sra_id}_2.fastq"

		echo "[*] Downloading ${sra_id}."

		# Prefetch
		echo "[*] Prefetching..."
		prefetch --output-directory $SRA_PREFETCH_DIR --progress --verify yes $sra_id

		# Fasterq-dump
		echo "[*] Invoking fasterq-dump..."
		fasterq-dump \
		--progress \
		--outdir . \
		--skip-technical \
		--print-read-nr \
		--force \
		-t ${FASTERQ_TMP_DIR} \
		"${SRA_PREFETCH_DIR}/${sra_id}/${sra_id}.sra"

		pigz -c ${sra_fq1} > ${gz1}
		pigz -c ${sra_fq2} > ${gz2}
	fi
}


run_trimmomatic()
{
	gz_file_1=$1
	gz_file_2=$2
	prefix=$3
	out_dir=$4

	# Target fastq files.
	mkdir -p $out_dir

	trimmed_1_paired="${out_dir}/${prefix}_paired_1.fastq"
	trimmed_2_paired="${out_dir}/${prefix}_paired_2.fastq"

	if [ -f "${trimmed_1_paired}" ] && [ -f "${trimmed_2_paired}" ]
	then
		echo "[*] Trimmomatic outputs already found!"
	else
		tmp_file_1="${out_dir}/${prefix}_1.fastq"
		tmp_file_2="${out_dir}/${prefix}_2.fastq"
		echo "[*] Decompressing to temporary output."
		pigz -dck ${gz_file_1} > $tmp_file_1
		pigz -dck ${gz_file_2} > $tmp_file_2

		echo "[*] Invoking kneaddata."
		kneaddata \
		--input1 ${gz_file_1} \
		--input2 ${gz_file_2} \
		--reference-db ${KNEADDATA_DB_DIR} \
		--output ${out_dir} \
		--trimmomatic-options "ILLUMINACLIP:${NEXTERA_ADAPTER_PATH}:2:30:10:2 LEADING:10 TRAILING:10 MINLEN:35" \
		--threads 8 \
		--quality-scores phred33 \
		--bypass-trf \
		--trimmomatic ${TRIMMOMATIC_DIR} \
		--output-prefix ${prefix}
	fi
}


# =============== Download background samples & preprocess them
cd ${BACKGROUND_FASTQ_DIR}
while IFS=, read -r tidx t sra_id num_background
do
	if [[ "$tidx" == "TIDX" ]]; then
		continue
	fi

	raw_sample_dir="raw"
	trimmomatic_outdir="trimmomatic/${sra_id}"
	mkdir -p $raw_sample_dir
	mkdir -p $trimmomatic_outdir

	if [ -f "${tidx}_background_1.fq" ] && [ -f "${tidx}_background_2.fq" ]
	then
		echo "[*] Background reads already found!"
	else
		raw_gz1=${raw_sample_dir}/${sra_id}_1.fastq.gz
		raw_gz2=${raw_sample_dir}/${sra_id}_2.fastq.gz
		download_sra $sra_id $raw_gz1 $raw_gz2
		run_trimmomatic $raw_gz1 $raw_gz2 $sra_id $trimmomatic_outdir

		trimmed_1_unpaired="${trimmomatic_outdir}/${sra_id}_unmatched_1.fastq"
		trimmed_1_paired="${trimmomatic_outdir}/${sra_id}_paired_1.fastq"
		trimmed_2_unpaired="${trimmomatic_outdir}/${sra_id}_unmatched_2.fastq"
		trimmed_2_paired="${trimmomatic_outdir}/${sra_id}_paired_2.fastq"
		cat $trimmed_1_unpaired $trimmed_1_paired > ${tidx}_background_1.fq
		cat $trimmed_2_unpaired $trimmed_2_paired > ${tidx}_background_2.fq
	fi
done < ${BACKGROUND_CSV}


# =============== Sample synthetic reads
for (( trial = 1; trial < ${N_TRIALS}+1; trial++ ));
do
	for n_reads in "${SYNTHETIC_COVERAGES[@]}"
	do
		seed=$((seed+1))

		trial_dir=$(get_trial_dir $n_reads $trial)
		read_dir=${trial_dir}/reads
		log_dir=${trial_dir}/logs

		if [[ -d "${read_dir}" ]]; then
			echo "[*] Skipping reads: ${n_reads}, trial #${trial}] -> ${trial_dir}"
		else
			echo "Sampling [Number of reads: ${n_reads}, trial #${trial}] -> ${trial_dir}"

			mkdir -p $log_dir
			mkdir -p $read_dir
			export CHRONOSTRAIN_LOG_FILEPATH="${log_dir}/read_sample.log"
			export CHRONOSTRAIN_CACHE_DIR="${trial_dir}/cache"

			python ${BASE_DIR}/helpers/sample_reads.py \
			--out_dir $read_dir \
			--abundance_path $RELATIVE_GROUND_TRUTH \
			--index_path ${REFSEQ_INDEX} \
			--num_reads $n_reads \
			--profiles $READ_PROFILE_PATH $READ_PROFILE_PATH \
			--read_len $READ_LEN \
			--seed ${seed} \
			--num_cores $N_CORES
		fi
	done
done