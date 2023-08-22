#!/bin/bash
set -e
source settings.sh

n_reads=$1
trial=$2

if [ -z "$n_reads" ]
then
	echo "var \"n_reads\" is empty"
	exit 1
fi

if [ -z "$trial" ]
then
	echo "var \"trial\" is empty"
	exit 1
fi

trial_dir=$(get_trial_dir $n_reads $trial)
read_dir=${trial_dir}/reads
output_dir=${trial_dir}/output/themisto
runtime_file=${trial_dir}/output/themisto_runtime.txt


if [ -f $runtime_file ]; then
	echo "[*] Skipping Themisto Pseudoalignment (n_reads: ${n_reads}, trial: ${trial})"
	exit 0
fi


echo "[*] Running Themisto pseudoalignment for n_reads: ${n_reads}, trial: ${trial}"
mkdir -p ${output_dir}


timepoint_sra_id()
{
  target_t_idx=$1
  while IFS=, read -r tidx t sra_id num_background
  do
    if [[ "$tidx" == "${target_t_idx}" ]]; then
      echo "${sra_id}"
    fi
  done < ${BACKGROUND_CSV}
}


echo "[**] Preparing themisto inputs..."
input_file=${output_dir}/query_files.txt
output_file=${output_dir}/output_files.txt
> $input_file
> $output_file

for t_idx in 0 1 2 3 4; do
	background_sra_id="$(timepoint_sra_id ${t_idx})"
	trimmomatic_outdir="${BACKGROUND_FASTQ_DIR}/trimmomatic/${background_sra_id}"

  echo "${read_dir}/${t_idx}_sim_1.fq" >> $input_file
	echo "${read_dir}/${t_idx}_sim_2.fq" >> $input_file
  echo "${trimmomatic_outdir}/${background_sra_id}_paired_1.fastq" >> $input_file
  echo "${trimmomatic_outdir}/${background_sra_id}_paired_2.fastq" >> $input_file

  echo "${output_dir}/${t_idx}_sim_1.output.txt" >> $output_file
	echo "${output_dir}/${t_idx}_sim_2.output.txt" >> $output_file
  echo "${output_dir}/${t_idx}_background_1.output.txt" >> $output_file
  echo "${output_dir}/${t_idx}_background_2.output.txt" >> $output_file
done


echo "[**] Running pseudoalignment."
start_time=$(date +%s%N)  # nanoseconds
tmp_dir="${output_dir}/_tmp"
cd ${THEMISTO_DB_DIR}
${THEMISTO_BIN_DIR}/themisto pseudoalign \
  --query-file-list $input_file \
  --out-file-list $output_file \
  --index-prefix enterobacteriaceae \
  --temp-dir $tmp_dir \
  --n-threads $N_CORES \
  --sort-output


# ====== Record runtime
end_time=$(date +%s%N)
elapsed_time=$(( $(($end_time-$start_time)) / 1000000 ))
echo "${elapsed_time}" > $runtime_file


# clean up.
echo '[**] Cleaning up...'
rm -rf $tmp_dir