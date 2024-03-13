#!/bin/bash
set -e
source settings.sh
source mgems/settings.sh


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
    bash mgems/run_pipeline_explicit.sh "${part_id}" "${sample_id}"
done < "${participant_dir}/dataset.tsv"
touch "${breadcrumb}"
