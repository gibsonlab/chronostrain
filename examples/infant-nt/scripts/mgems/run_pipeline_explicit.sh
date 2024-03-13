#!/bin/bash
set -e
source settings.sh
source mgems/settings.sh
shopt -s nullglob

# demix_check should point to demix_check.py (https://github.com/harry-thorpe/demix_check).
# To pass this first check, create a bash executable called "demix_check" that invokes `python demix_check.py` and add to PATH environment var.
require_program demix_check
require_program themisto
require_program mSWEEP
require_program mGEMS
require_program gzip

# ============ Requires arguments:
participant=$1
sample_id=$2
require_variable 'participant' $participant
require_variable 'sample_id' $sample_id

workdir=$(pwd)
participant_dir=${DATA_DIR}/${participant}
output_dir=${participant_dir}/mgems/${sample_id}
breadcrumb=${output_dir}/mgems.${sample_id}.DONE
if [ -f ${breadcrumb} ]; then
  echo "[*] mGEMS hierarchical pipeline for ${participant} [Sample ${sample_id}] already done."
  exit 0
fi

echo "[*] Running mGEMS hierarchical pipeline for ${participant}, sample ${sample_id}"
mkdir -p "${output_dir}"
#fq_1=${participant_dir}/reads/${sample_id}_kneaddata/${sample_id}_paired_1.fastq.gz
#fq_2=${participant_dir}/reads/${sample_id}_kneaddata/${sample_id}_paired_2.fastq.gz
fq_1=${participant_dir}/reads/${sample_id}_1.fastq.gz
fq_2=${participant_dir}/reads/${sample_id}_2.fastq.gz
if ! [ -f ${fq_1} ]; then
  echo "Forward read not found (Expected: ${fq_1})"
  exit 1
fi
if ! [ -f ${fq_2} ]; then
  echo "Reverse read not found (Expected: ${fq_2})"
  exit 1
fi

# ========= chdir so indices are relative-pathable
aln_and_compress()
{
	fq1=$1
	fq2=$2
	aln_out1=$3
	aln_out2=$4
	ref_idx=$5
	n_colors=$6
	tmp_dir=$7

  mkdir -p ${tmp_dir}
  input_file=${tmp_dir}/query_files.txt
  output_file=${tmp_dir}/output_files.txt
	aln_raw1=${tmp_dir}/aln1_raw.txt
	aln_raw2=${tmp_dir}/aln2_raw.txt

	# prepare input txt file list
  echo "${fq1}" > "$input_file"
  echo "${fq2}" >> "$input_file"
  echo "${aln_raw1}" > "$output_file"
  echo "${aln_raw2}" >> "$output_file"
	themisto pseudoalign \
    --index-prefix ${ref_idx} --rc --temp-dir ${tmp_dir} --n-threads ${N_CORES} --sort-output-lines \
    --query-file-list "$input_file" \
    --out-file-list "$output_file" \

  n_reads1=$(wc -l < "${aln_raw1}")
  n_reads2=$(wc -l < "${aln_raw2}")
  if [[ $n_reads1 != $n_reads2 ]]; then
    echo "# of reads in pseudoalignments don't match (${n_reads1} vs ${n_reads2})."
    exit 1
  fi
  mv ${aln_raw1} ${aln_out1}
  mv ${aln_raw2} ${aln_out2}
#  alignment-writer -n $n_colors -r $n_reads1 -f $aln_raw1 > $aln_out1
#  alignment-writer -n $n_colors -r $n_reads2 -f $aln_raw2 > $aln_out2
}

# ============================================ species-level analysis\
species_outdir=${output_dir}/species
aln_1=${species_outdir}/ali_1.aln
aln_2=${species_outdir}/ali_2.aln
mkdir -p ${species_outdir}

cd ${SPECIES_REF_DIR}
echo "[*] Species-level analysis (Refdir=${SPECIES_REF_DIR})."
echo "[**] Aligning fwd+rev reads"
aln_and_compress ${fq_1} ${fq_2} ${aln_1} ${aln_2} ${SPECIES_REF_INDEX} ${SPECIES_N_COLORS} ${species_outdir}/tmp

echo "[**] Cleaning up alignment tmpdir."
rm -rf ${species_outdir}/tmp

echo "[**] Running mSWEEP abundance estimation."
mSWEEP \
  -t ${N_CORES} \
  --themisto-1 ${aln_1}  \
  --themisto-2 ${aln_2}  \
  -o ${species_outdir}/msweep \
  -i ${SPECIES_REF_CLUSTER} \
  --bin-reads \
  --target-groups "Enterococcus_faecalis" \
  --verbose


echo "[**] Running mGEMS extract."
mkdir -p ${species_outdir}/binned_reads
mGEMS extract \
  --bins ${species_outdir}/Enterococcus_faecalis.bin \
  -r ${fq_1},${fq_2} \
  -o ${species_outdir}/binned_reads
for f in ${species_outdir}/binned_reads/*.fastq; do gzip "$f"; done


# ============================================ strain-level analysis
echo "[*] Strain-level analysis."
strain_outdir=${output_dir}/Efaecalis
strain_fq_1=${species_outdir}/binned_reads/Enterococcus_faecalis_1.fastq.gz
strain_fq_2=${species_outdir}/binned_reads/Enterococcus_faecalis_2.fastq.gz
strain_aln_1=${strain_outdir}/ali_1.aln
strain_aln_2=${strain_outdir}/ali_2.aln
mkdir -p ${strain_outdir}

cd ${EFAECALIS_REF_DIR}
echo "[**] Aligning fwd reads"
aln_and_compress ${strain_fq_1} ${strain_fq_2} ${strain_aln_1} ${strain_aln_2} ${EFAECALIS_REF_INDEX} ${EFAECALIS_N_COLORS} ${strain_outdir}/tmp

echo "[**] Cleaning up alignment tmpdir."
rm -rf ${strain_outdir}/tmp

echo "[**] Running mSWEEP abundance estimation."
mSWEEP \
  -t ${N_CORES} \
  --themisto-1 ${strain_aln_1}  \
  --themisto-2 ${strain_aln_2}  \
  -o ${strain_outdir}/msweep \
  -i ${EFAECALIS_REF_CLUSTER} \
  --bin-reads \
  --min-abundance 0.01 \
  --verbose


echo "[**] Extracting reads (for demix_check)."
mkdir -p ${strain_outdir}/binned_reads
for bin_file in ${strain_outdir}/*.bin; do
  mv ${bin_file} ${strain_outdir}/binned_reads
done

for bin_file in ${strain_outdir}/binned_reads/*.bin; do
  mGEMS extract --bins ${bin_file} -r ${strain_fq_1},${strain_fq_2} -o ${strain_outdir}/binned_reads
done

echo "[**] Compressing extracted reads."
for f in ${strain_outdir}/binned_reads/*.fastq; do gzip "$f"; done

echo "[**] Running demix_check."
demix_check --mode_check \
  --binned_reads_dir ${strain_outdir}/binned_reads \
  --msweep_abun ${strain_outdir}/msweep_abundances.txt \
  --out_dir ${strain_outdir}/demix_check \
  --ref ${EFAECALIS_REF_INDEX}/demix_check_index

cd ${workdir}
touch "${breadcrumb}"
