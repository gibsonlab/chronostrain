#!/bin/bash
set -e
source settings.sh

seed=0
check_program 'art_illumina'
check_program 'prefetch'
check_program 'fasterq-dump'
check_program 'seqtk'

# First, download the background sample.
if [[ -f $BACKGROUND_FQ_1 && -f $BACKGROUND_FQ_2 ]]; then
	echo "[*] Target files for ${BACKGROUND_SRA_ID} already exist."
else
	mkdir -p ${BACKGROUND_FASTQ_DIR}
	cd ${BACKGROUND_FASTQ_DIR}

	# Prefetch
	echo "[*] Prefetching..."
	prefetch --output-directory $SRA_PREFETCH_DIR --progress --verify yes $BACKGROUND_SRA_ID

	# Fasterq-dump
	echo "[*] Invoking fasterq-dump..."
	fasterq-dump \
	--progress \
	--outdir . \
	--skip-technical \
	--print-read-nr \
	--force \
	-t ${FASTERQ_TMP_DIR} \
	"${SRA_PREFETCH_DIR}/${BACKGROUND_SRA_ID}/${BACKGROUND_SRA_ID}.sra"
fi


for n_reads in 10000 25000 50000 75000 100000
do
	for (( trial = 1; trial < ${N_TRIALS}+1; trial++ ));
	do
		seed=$((seed+1))

		trial_dir=$(get_trial_dir $n_reads $trial)
		read_dir=${trial_dir}/reads
		log_dir=${trial_dir}/logs

		echo "[Number of reads: ${n_reads}, trial #${trial}] -> ${trial_dir}"

		mkdir -p $log_dir
		mkdir -p $read_dir
		export CHRONOSTRAIN_LOG_FILEPATH="${log_dir}/read_sample.log"
		export CHRONOSTRAIN_CACHE_DIR="${trial_dir}/cache"

		python ${BASE_DIR}/helpers/sample_reads.py \
		--out_dir $read_dir \
		--abundance_path $RELATIVE_GROUND_TRUTH \
		--index_path ${REFSEQ_INDEX} \
		--background_samples $BACKGROUND_FQ_1 $BACKGROUND_FQ_2 \
		--background_read_counts ${BACKGROUND_N_READS} \
		--num_reads $n_reads \
		--profiles $READ_PROFILE_PATH $READ_PROFILE_PATH \
		--read_len $READ_LEN \
		--seed ${seed} \
		--num_cores $N_CORES
	done
done