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

cd ${DEMIX_REF_DIR}
demix_check --mode_run --r1 "${fq_1}" --r2 "${fq_2}" --out_dir "${output_dir}" --ref "${DEMIX_REF_HIERARCHICAL}" --threads ${N_CORES}
cd -
touch "${breadcrumb}"
