#!/bin/bash
set -e
source settings.sh
source mgems/settings.sh
shopt -s nullglob

# demix_check should point to demix_check.py (https://github.com/harry-thorpe/demix_check).
require_program themisto
require_program mSWEEP
require_program alignment-writer

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
	fq_in=$1
	aln_out=$2
	tmp_dir=$3

	aln_raw=${aln_out}-raw.txt
	n_ref=$(wc -l < "ref_clu.txt")
	themisto pseudoalign \
    --index-prefix ${ECOLI_REF_INDEX} --rc --temp-dir ${tmp_dir} --n-threads ${N_CORES} --sort-output-lines \
    --query-file ${fq_in} \
    --outfile ${aln_raw}

  n_reads=$(wc -l < "${aln_raw}")
  alignment-writer -n $n_ref -r $n_reads -f $aln_raw > $aln_out
  rm ${aln_raw}
}

# =================== paths
trial_dir=$(get_trial_dir "${mutation_ratio}" "$replicate" "$n_reads" "$trial")
output_dir=${trial_dir}/output/${subdir}/${time_point}/Ecoli
species_output_dir=${trial_dir}/output/${subdir}/${time_point}/species
runtime_file=${output_dir}/runtime.${time_point}.txt
mkdir -p $output_dir


if [ -f $runtime_file ]; then
	echo "[*] Skipping strain-level analysis (mut_ratio: ${mutation_ratio} | replicate: ${replicate} |  n_reads: ${n_reads} | trial: ${trial} | timepoint ${time_point})"
	exit 0
fi

# =================== input files
fq_1=${species_output_dir}/binned_reads/Ecoli_1.fastq.gz
fq_2=${species_output_dir}/binned_reads/Ecoli_2.fastq.gz

# ===================================== ENSURE THIS IS SET TO THE RIGHT DATABASE REFDIR!
cd ${ECOLI_REF_DIR}
echo "[*] Refdir = ${ECOLI_REF_DIR}, Index= ${ECOLI_REF_INDEX}"

# =================== Analysis starts here
start_time=$(date +%s%N)  # nanoseconds
tmp_dir="${output_dir}/_tmp"
mkdir -p ${tmp_dir}

echo "[*] Strain-level analysis."
aln_1=${output_dir}/aln_1.aln
aln_2=${output_dir}/aln_2.aln

echo "[**] Aligning fwd reads"
aln_and_compress "${fq_1}" "${aln_1}" "${output_dir}/_tmp"
echo "[**] Aligning rev reads"
aln_and_compress "${fq_2}" "${aln_2}" "${output_dir}/_tmp"

echo "[**] Running mSWEEP abundance estimation."
mSWEEP \
  -t ${N_CORES} \
  --themisto-1 ${aln_1}  \
  --themisto-2 ${aln_2}  \
  -o ${output_dir}/msweep \
  -i ${ECOLI_REF_CLUSTER} \
  -q ${beta_binomial_mean}


# ====== Record runtime (don't include read extraction/demix-check. include core alg only)
end_time=$(date +%s%N)
elapsed_time=$(( $(($end_time-$start_time)) / 1000000 ))
echo "${elapsed_time}" > $runtime_file
