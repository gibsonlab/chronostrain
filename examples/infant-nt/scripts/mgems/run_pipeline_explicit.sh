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


participant_dir=${DATA_DIR}/${participant}
output_dir=${participant_dir}/mgems/${sample_id}
breadcrumb=${output_dir}/mgems.${sample_id}.DONE
if [ -f ${breadcrumb} ]; then
  echo "[*] mGEMS hierarchical pipeline for ${participant} [Sample ${sample_id}] already done."
  exit 0
fi

echo "[*] Running mGEMS hierarchical pipeline for ${participant}, sample ${sample_id}"
mkdir -p "${output_dir}"
fq_1=${participant_dir}/reads/${sample_id}_kneaddata/${sample_id}_paired_1.fastq.gz
fq_2=${participant_dir}/reads/${sample_id}_kneaddata/${sample_id}_paired_2.fastq.gz
if ! [ -f ${fq_1} ]; then
  echo "Forward read not found (Expected: ${fq_1})"
  exit 1
fi
if ! [ -f ${fq_2} ]; then
  echo "Reverse read not found (Expected: ${fq_2})"
  exit 1
fi

# ========= chdir so indices are relative-pathable
echo "[*] Work dir: ${DEMIX_REF_DIR}"
cd ${DEMIX_REF_DIR}

aln_and_compress()
{
	fq_in=$1
	aln_out=$2
	refdir=$3
	tmp_dir=$4

	aln_raw=${aln_out}-raw.txt
	n_ref=$(wc -l < "${refdir}/ref_clu.txt")
	themisto pseudoalign \
    --index-prefix ${refdir}/ref_idx/ref_idx --rc --temp-dir ${tmp_dir} --n-threads ${N_CORES} --sort-output-lines \
    --query-file ${fq_in} \
    --outfile ${aln_raw}

  n_reads=$(wc -l < "${aln_raw}")
  alignment-writer -n $n_ref -r $n_reads -f $aln_raw > $aln_out
  rm ${aln_raw}
}

# ============================================ species-level analysis
species_refdir=ref_dir/species_ref
species_outdir=${output_dir}/species_ref
aln_1=${species_outdir}/ali_1.aln
aln_2=${species_outdir}/ali_2.aln
mkdir -p ${species_outdir}

echo "[*] Species-level analysis."
echo "[**] Aligning fwd reads"
aln_and_compress ${fq_1} ${aln_1} ${species_refdir} ${species_outdir}/tmp
echo "[**] Aligning rev reads"
aln_and_compress ${fq_2} ${aln_2} ${species_refdir} ${species_outdir}/tmp

echo "[**] Running mSWEEP abundance estimation."
mSWEEP \
  -t ${N_CORES} \
  --themisto-1 ${aln_1}  \
  --themisto-2 ${aln_2}  \
  -o ${species_outdir}/msweep \
  -i ${species_refdir}/ref_clu.txt \
  --bin-reads \
  --target-groups Efaecalis \
  --verbose


echo "[**] Running mGEMS extract."
mkdir -p ${species_outdir}/binned_reads
mGEMS extract \
  --bins ${species_outdir}/Efaecalis.bin \
  -r ${fq_1},${fq_2} \
  -o ${species_outdir}/binned_reads
for f in ${species_outdir}/binned_reads/*.fastq; do gzip "$f"; done


# ============================================ strain-level analysis
echo "[*] Strain-level analysis."
strain_refdir=ref_dir/Efaecalis
strain_outdir=${output_dir}/Efaecalis
strain_fq_1=${species_outdir}/binned_reads/Efaecalis_1.fastq.gz
strain_fq_2=${species_outdir}/binned_reads/Efaecalis_2.fastq.gz
strain_aln_1=${strain_outdir}/ali_1.aln
strain_aln_2=${strain_outdir}/ali_2.aln
mkdir -p ${strain_outdir}

echo "[**] Aligning fwd reads"
aln_and_compress ${strain_fq_1} ${strain_aln_1} ${strain_refdir} ${strain_outdir}/tmp
echo "[**] Aligning rev reads"
aln_and_compress ${strain_fq_2} ${strain_aln_2} ${strain_refdir} ${strain_outdir}/tmp


echo "[**] Running mSWEEP abundance estimation."
mSWEEP \
  -t ${N_CORES} \
  --themisto-1 ${strain_aln_1}  \
  --themisto-2 ${strain_aln_2}  \
  -o ${strain_outdir}/msweep \
  -i ${strain_refdir}/ref_clu.txt \
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
demix_check_file="${strain_outdir}/demix_check.tsv"
> $demix_check_file

for bin_file in ${strain_outdir}/binned_reads/*.bin; do
  bin_id="$(basename ${bin_file} .bin)"
  echo "[***] Checking bin id = ${bin_id}"
  check_reads.sh \
    --cluster "${bin_id}" \
    --abundances ${strain_outdir}/msweep_abundances.txt \
    --threads ${N_CORES} \
    --tmpdir ${strain_outdir}/tmp \
    --fwd ${strain_outdir}/binned_reads/${bin_id}_1.fastq.gz \
    --rev ${strain_outdir}/binned_reads/${bin_id}_2.fastq.gz \
    --reference ${strain_refdir} \
    >> ${demix_check_file}
done
##demix_check --mode_check \
##  --binned_reads_dir ${strain_outdir}/binned_reads \
##  --msweep_abun ${strain_outdir}/msweep_abundances.txt \
##  --out_dir ${strain_outdir}/demix_check \
##  --ref ${strain_refdir}

cd -
touch "${breadcrumb}"
