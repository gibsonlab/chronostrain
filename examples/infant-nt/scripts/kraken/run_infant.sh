#!/bin/bash
source settings.sh
set -e

infant_id=$1
infant_dir=${DATA_DIR}/${infant_id}

while IFS=$'\t' read part_id time_point sample_id read1_raw_fq read2_raw_fq
do
    if [ "${part_id}" == "Participant" ]; then continue; fi
    bash kraken/run_sample.sh ${infant_id} ${sample_id}
done < "${infant_dir}/dataset.tsv"
