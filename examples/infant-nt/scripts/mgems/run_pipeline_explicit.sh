#!/bin/bash
set -e
source settings.sh
source mgems/settings.sh

# demix_check should point to demix_check.py (https://github.com/harry-thorpe/demix_check).
# To pass this first check, create a bash executable called "demix_check" that invokes `python demix_check.py` and add to PATH environment var.
require_program demix_check
require_program themisto
require_program mSWEEP
require_program mGEMS

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
echo "[*] Work dir: ${DEMIX_REF_DIR}"
cd ${DEMIX_REF_DIR}

# ============================================ species-level analysis
species_refdir=ref_dir/species_ref
species_outdir=${output_dir}/species_ref
aln_1=${species_outdir}/ali_1.aln
aln_2=${species_outdir}/ali_2.aln
mkdir -p ${species_outdir}

echo "[*] Species-level analysis."
echo "[**] Aligning fwd reads"

themisto pseudoalign \
  --index-prefix ${species_refdir}/ref_idx/ref_idx --rc --temp-dir ${species_outdir}/tmp --n-threads ${N_CORES} --sort-output --gzip-output \
  --query-file ${fq_1} \
  --outfile ${aln_1}
echo "[**] Aligning rev reads"
themisto pseudoalign \
  --index-prefix ${species_refdir}/ref_idx/ref_idx --rc --temp-dir ${species_outdir}/tmp --n-threads ${N_CORES} --sort-output --gzip-output \
  --query-file ${fq_2} \
  --outfile ${aln_2}

echo "[**] Running mSWEEP abundance estimation."
mSWEEP \
  -t ${N_CORES} \
  --themisto-1 ${aln_1}.gz  \
  --themisto-2 ${aln_2}.gz  \
  -o ${species_outdir}/msweep \
  -i ${species_refdir}/ref_clu.txt \
  --write-probs \
  --bin-reads \
  --target-groups Efaecalis \
  --verbose

### use mSWEEP built-in mgems binning impl (mSWEEP --bin-reads), instead of below
##echo "[**] Running mGEMS binning."
mkdir -p ${species_outdir}/binned_reads
#mGEMS bin \
##  --groups Efaecalis \
##  --themisto-alns ${aln_1}.gz,${aln_2}.gz \
##  -o ${species_outdir}/binned_reads \
##  --probs ${species_outdir}/msweep_probs.tsv.gz \
##  -a ${species_outdir}/msweep_abundances.txt \
##  --index ${species_refdir}/ref_idx \
##  -i ${species_refdir}/ref_clu.txt
##  --min-abundance 0.01 \  # note: this makes the pipeline miss Efaecalis for many samples.

echo "[**] Extracting binned reads."
mGEMS extract \
  --bins ${species_outdir}/Efaecalis.bin \
  -r ${fq_1},${fq_2} \
  -o ${species_outdir}/binned_reads


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
themisto pseudoalign \
  --index-prefix ${strain_refdir}/ref_idx/ref_idx --rc --temp-dir ${strain_outdir}/tmp --n-threads ${N_CORES} --sort-output --gzip-output \
  --query-file ${strain_fq_1} \
  --outfile ${strain_aln_1}
echo "[**] Aligning rev reads"
themisto pseudoalign \
  --index-prefix ${strain_refdir}/ref_idx/ref_idx --rc --temp-dir ${strain_outdir}/tmp --n-threads ${N_CORES} --sort-output --gzip-output \
  --query-file ${strain_fq_2} \
  --outfile ${strain_aln_2}

echo "[**] Running mSWEEP abundance estimation."
mSWEEP \
  -t ${N_CORES} \
  --themisto-1 ${strain_aln_1}.gz  \
  --themisto-2 ${strain_aln_2}.gz  \
  -o ${strain_outdir}/msweep \
  -i ${strain_refdir}/ref_clu.txt \
  --write-probs \
  --verbose

cd -
touch "${breadcrumb}"
