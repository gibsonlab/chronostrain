#!/bin/bash
set -e
source settings.sh
source mgems/settings.sh
shopt -s nullglob

require_program themisto
require_program mSWEEP
require_program mGEMS
require_program alignment-writer
require_program pigz

# ============ Requires arguments:
mutation_ratio=$1
replicate=$2
n_reads=$3
trial=$4
time_point=$5
subdir=$6
beta_binomial_mean=$7

require_variable "mutation_ratio" $mutation_ratio
require_variable "replicate" $replicate
require_variable "n_reads" $n_reads
require_variable "trial" $trial
require_variable "time_point" $time_point
require_variable "subdir" $subdir
require_variable "beta_binomial_mean" $beta_binomial_mean


# =================== helper function
aln_and_compress()
{
	in1=$1
	in2=$2
	aln1=$3
	aln2=$4
	tmp_dir=$5

	input_file=${tmp_dir}/query_files.txt
  output_file=${tmp_dir}/output_files.txt
	aln_raw1=${tmp_dir}/aln1.txt
	aln_raw2=${tmp_dir}/aln2.txt

	# prepare input txt file list
  echo "${in1}" > "$input_file"
  echo "${in2}" >> "$input_file"

  echo "${aln_raw1}" > "$output_file"
  echo "${aln_raw2}" >> "$output_file"

	themisto pseudoalign \
    --index-prefix ${SPECIES_REF_INDEX} --rc --temp-dir ${tmp_dir} --n-threads ${N_CORES} --sort-output-lines \
    --query-file-list "$input_file" \
    --out-file-list "$output_file" \

  n1=$(wc -l < "${aln_raw1}")
  n2=$(wc -l < "${aln_raw2}")
  echo "alignment-writer -n ${SPECIES_N_COLORS} -r $n1 -f $aln_raw1 > $aln1"
  alignment-writer -n ${SPECIES_N_COLORS} -r $n1 -f $aln_raw1 > $aln1
  echo "alignment-writer -n ${SPECIES_N_COLORS} -r $n2 -f $aln_raw2 > $aln2"
  alignment-writer -n ${SPECIES_N_COLORS} -r $n2 -f $aln_raw2 > $aln2
}

# =================== paths
trial_dir=$(get_trial_dir "${mutation_ratio}" "$replicate" "$n_reads" "$trial")
output_dir=${trial_dir}/output/${subdir}/${time_point}/species
read_dir=${trial_dir}/reads
runtime_file=${output_dir}/runtime.${time_point}.txt
mkdir -p $output_dir


if [ -f $runtime_file ]; then
	echo "[*] Skipping species-binning analysis (mut_ratio: ${mutation_ratio} | replicate: ${replicate} |  n_reads: ${n_reads} | trial: ${trial} | timepoint ${time_point})"
	exit 0
fi

# ===================================== ENSURE THIS IS SET TO THE RIGHT DATABASE REFDIR!
cd ${SPECIES_REF_DIR}
echo "[*] Refdir = ${SPECIES_REF_DIR}, Index= ${SPECIES_REF_INDEX}"

# =================== input files
fq_sim_1="${read_dir}/${time_point}_sim_1.fq"
fq_bg_1="${BACKGROUND_FASTQ_DIR}/sorted/${time_point}_background_1.sorted.fq"
fq_sim_2="${read_dir}/${time_point}_sim_2.fq"
fq_bg_2="${BACKGROUND_FASTQ_DIR}/sorted/${time_point}_background_2.sorted.fq"

# =================== Concatenate input fastq files (for extraction using mgems)
echo "[*] Concatenating input fastQ files (for binning)"
fq_1=${output_dir}/1.fq
fq_2=${output_dir}/2.fq
cat ${fq_sim_1} ${fq_bg_1} > ${fq_1}
cat ${fq_sim_2} ${fq_bg_2} > ${fq_2}

# =================== Analysis starts here
start_time=$(date +%s%N)  # nanoseconds
tmp_dir="${output_dir}/_tmp"
mkdir -p ${tmp_dir}

echo "[*] Species-level analysis."
aln_1=${output_dir}/aln_1.aln
aln_2=${output_dir}/aln_2.aln

echo "[**] Aligning reads"
aln_and_compress "${fq_1}" "${fq_2}" "${aln_1}" "${aln_2}" "${tmp_dir}"

echo "[**] Running mSWEEP abundance estimation."
mSWEEP \
  -t ${N_CORES} \
  --themisto-1 ${aln_1}  \
  --themisto-2 ${aln_2}  \
  -o ${output_dir}/msweep \
  -i ${SPECIES_REF_CLUSTER} \
  --bin-reads \
  --target-groups "Escherichia_coli" \
  --min-abundance 0.0 \
  -q ${beta_binomial_mean} \
  --verbose



echo "[**] Running mGEMS extract."

mkdir -p ${output_dir}/binned_reads
mv ${output_dir}/Escherichia_coli.bin ${output_dir}/binned_reads/Ecoli.bin

cd ${output_dir}  # for some reason, mGEMS extract only works if you cd into the directory.
mGEMS extract \
  --bins binned_reads/Ecoli.bin \
  -r ${fq_1},${fq_2} \
  -o binned_reads
cd -


# ====== Record runtime
end_time=$(date +%s%N)
elapsed_time=$(( $(($end_time-$start_time)) / 1000000 ))
echo "${elapsed_time}" > $runtime_file


# Just in case the files weren't gzipped properly.
for f in ${output_dir}/binned_reads/*.fastq; do pigz "$f"; done

# clean up.
echo '[**] Cleaning up...'
rm -rf ${tmp_dir}
rm $fq_1
rm $fq_2
