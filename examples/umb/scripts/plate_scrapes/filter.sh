#!/bin/bash
set -e

source settings.sh

# =========== Read filtering. ===============
for umb_plate_dir in ${OUTPUT_DIR}/*
do
  sample_name="$(basename $umb_plate_dir)"
  breadcrumb=${umb_plate_dir}/filter.DONE
	export CHRONOSTRAIN_LOG_FILEPATH="${umb_plate_dir}/filter.log"

  if [ -f $breadcrumb ]
	then
	  echo "Skipping filter for ${umb_id}."
	else
    echo "Filtering reads for ${sample_name}"
    env JAX_PLATFORM_NAME=cpu chronostrain filter \
      -r "${umb_plate_dir}/reads.csv" \
      -o "${umb_plate_dir}/filtered" \
      --aligner "bwa-mem2"

	  touch $breadcrumb
	fi
done
