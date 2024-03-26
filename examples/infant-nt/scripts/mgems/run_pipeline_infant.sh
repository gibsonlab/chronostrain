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
require_variable 'participant' $participant


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
    bash mgems/bin_sample_species.sh "${part_id}" "${sample_id}"
    bash mgems/quantify_efaecalis.sh "${part_id}" "${sample_id}"
    bash mgems/quantify_efaecalis_chronostrain_mirror.sh "${part_id}" "${sample_id}"
    #bash mgems/quantify_efaecalis_chronostrain_mirror_99_99pct.sh "${part_id}" "${sample_id}"
    bash mgems/quantify_efaecalis_chronostrain_mutation_0002.sh "${part_id}" "${sample_id}"
done < "${participant_dir}/dataset.tsv"
touch "${breadcrumb}"
