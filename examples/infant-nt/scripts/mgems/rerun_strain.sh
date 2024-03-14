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
if ! [ -f ${breadcrumb} ]; then
  echo "[*] mGEMS hierarchical pipeline for ${participant} [Sample ${sample_id}] not yet done."
  exit 0
fi

echo "[*] Re-running strain-level inference for ${participant}, sample ${sample_id}"
mkdir -p "${output_dir}"


# ============================================ strain-level analysis
echo "[*] Strain-level analysis."
strain_outdir=${output_dir}/Efaecalis
species_outdir=${output_dir}/species
strain_fq_1=${species_outdir}/binned_reads/Enterococcus_faecalis_1.fastq.gz
strain_fq_2=${species_outdir}/binned_reads/Enterococcus_faecalis_2.fastq.gz
strain_aln_1=${strain_outdir}/ali_1.aln
strain_aln_2=${strain_outdir}/ali_2.aln
mkdir -p ${strain_outdir}

cd ${EFAECALIS_REF_DIR}

echo "[**] Running mSWEEP abundance estimation (min abundance = 0.0)."
mSWEEP \
  -t ${N_CORES} \
  --themisto-1 ${strain_aln_1}  \
  --themisto-2 ${strain_aln_2}  \
  -o ${strain_outdir}/msweep \
  -i ${EFAECALIS_REF_CLUSTER} \
  --bin-reads \
  --min-abundance 0.0 \
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
  --ref ./demix_check_index

cd ${workdir}
touch "${breadcrumb}"

breadcrumb=${output_dir}/mgems.${sample_id}.RERUN.DONE
touch $breadcrumb
