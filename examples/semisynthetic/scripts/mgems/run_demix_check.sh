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


# =================== paths
trial_dir=$(get_trial_dir "${mutation_ratio}" "$replicate" "$n_reads" "$trial")
species_output_dir=${trial_dir}/output/${subdir}/${time_point}/species
output_dir=${trial_dir}/output/${subdir}/${time_point}/Ecoli
mkdir -p $output_dir

breadcrumb=${output_dir}/demix_check.DONE
if [ -f ${breadcrumb} ]; then
  echo "[*] Demix_check already done."
  exit 0
fi


# ===================================== ENSURE THIS IS SET TO THE RIGHT DATABASE REFDIR!
cd ${ECOLI_REF_DIR}
echo "[*] Refdir = ${ECOLI_REF_DIR}, Index= ${ECOLI_REF_INDEX}"

# =================== input files
fq_1=${species_output_dir}/binned_reads/Ecoli_1.fastq.gz
fq_2=${species_output_dir}/binned_reads/Ecoli_2.fastq.gz

# =================== Analysis starts here
echo "[*] Second-round strain analysis for demix_check."
aln_1=${output_dir}/aln_1.aln
aln_2=${output_dir}/aln_2.aln

if ! [ -f ${aln_1} ]; then
  echo "Forward-read alignment ${aln_1} does not exist; cannot run demix_check."
  exit 1
fi
if ! [ -f ${aln_2} ]; then
  echo "Reverse-read alignment ${aln_2} does not exist; cannot run demix_check."
  exit 1
fi

echo "[**] Re-running mSWEEP abundance estimation (for read binning)"
mSWEEP \
  -t ${N_CORES} \
  --themisto-1 ${aln_1}  \
  --themisto-2 ${aln_2}  \
  -o ${output_dir}/msweep \
  -i ${ECOLI_REF_CLUSTER} \
  -q ${beta_binomial_mean} \
  --bin-reads \
  --min-abundance 0.0


echo "[**] Extracting reads (for demix_check)."
mkdir -p ${output_dir}/binned_reads
for bin_file in ${output_dir}/*.bin; do
  mv ${bin_file} ${output_dir}/binned_reads
done


cd ${output_dir}
for bin_file in binned_reads/*.bin; do
  echo "mGEMS extract --bins ${bin_file} -r ${fq_1},${fq_2} -o binned_reads"
  mGEMS extract --bins ${bin_file} -r ${fq_1},${fq_2} -o binned_reads
done

echo "[**] Compressing extracted reads."
for f in binned_reads/*.fastq; do gzip "$f"; done

demix_check --mode_check \
  --binned_reads_dir binned_reads \
  --msweep_abun msweep_abundances.txt \
  --out_dir demix_check \
  --ref ${ECOLI_REF_DIR} \
  --min_abun 0.0 \
  --threads ${N_CORES}
touch ${breadcrumb}
