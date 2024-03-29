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
require_program gzip

# ============ Requires arguments:
participant=$1
mutation_rate=$2
require_variable 'participant' $participant
require_variable 'mutation_rate' $mutation_rate


participant_dir=${DATA_DIR}/${participant}
output_dir=${participant_dir}/mgems
breadcrumb=${output_dir}/mgems.DONE
if [ -f "${breadcrumb}" ]; then
  echo "[*] mGEMS hierarchical pipeline for ${participant} <all samples> already done."
  exit 0
fi


mkdir -p "${output_dir}"
while IFS=$'\t' read part_id time_point sample_id read1_raw_fq read2_raw_fq
do
    if [ "${part_id}" == "Participant" ]; then continue; fi
    bash mutations/mgems_quantify_efaecalis_chronostrain_mutation.sh "${part_id}" "${sample_id}" "${mutation_rate}"
done < "${participant_dir}/dataset.tsv"
touch "${breadcrumb}"
