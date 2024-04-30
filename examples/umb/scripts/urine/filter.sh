#!/bin/bash
set -e

source settings.sh
export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/filter.log"

# =========== Read filtering. ===============

for umb_id in UMB01 UMB02 UMB03 UMB04 UMB05 UMB06 UMB07 UMB08 UMB09 UMB10 UMB11 UMB12 UMB13 UMB14 UMB15 UMB16 UMB17 UMB18 UMB19 UMB20 UMB21 UMB22 UMB23 UMB24 UMB25 UMB26 UMB27 UMB28 UMB29 UMB30 UMB31
do
  run_dir=${OUTPUT_DIR}/${umb_id}
  breadcrumb=${run_dir}/filter.DONE
	export CHRONOSTRAIN_LOG_FILEPATH="${run_dir}/logs/chronostrain_filter.log"
	reads_file="${run_dir}/reads.csv"
	echo "$breadcrumb"

  if ! [ -f $reads_file ]; then
    echo "reads file not found for ${umb_id}."
  elif [ -f $breadcrumb ]; then
	  echo "Skipping filter for ${umb_id}."
	else
    echo "[*] Filtering reads for ${umb_id}"
    env JAX_PLATFORM_NAME=cpu chronostrain filter \
      -r $reads_file \
      -o "${run_dir}/filtered" \
      -s ${CHRONOSTRAIN_CLUSTER_FILE} \
      --aligner "bwa-mem2"

	  touch $breadcrumb
	fi
done
