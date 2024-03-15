#!/bin/bash
set -e
source settings.sh
source mgems/settings.sh
shopt -s nullglob

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
breadcrumb=${output_dir}/mgems.species.DONE
if [ -f ${breadcrumb} ]; then
  echo "[*] mGEMS species-binning for ${participant} [Sample ${sample_id}] already done."
  exit 0
fi

# ====================================================== script begins here
echo "[*] Running mGEMS species-binning for ${participant}, sample ${sample_id}"
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


cd ${workdir}
touch "${breadcrumb}"
