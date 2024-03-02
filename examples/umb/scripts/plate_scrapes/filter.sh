#!/bin/bash
set -e

source settings.sh

# =========== Read filtering. ===============
for umb_plate_dir in ${OUTPUT_DIR}/UMB18*/UMB*; do
  sample_name="$(basename $umb_plate_dir)"
  echo "[*] Found sample dir ${umb_plate_dir}"

  filter_breadcrumb=${umb_plate_dir}/filter.DONE
  inference_breadcrumb=${umb_plate_dir}/inference.DONE

  if [ -f ${filter_breadcrumb} ]; then
    echo "[*] Skipping filter for ${sample_name}"
  else
    echo "[*] Filtering reads for ${sample_name}"
    export CHRONOSTRAIN_LOG_FILEPATH="${umb_plate_dir}/filter.log"
    env JAX_PLATFORM_NAME=cpu chronostrain filter \
      -r "${umb_plate_dir}/reads.csv" \
      -o "${umb_plate_dir}/filtered" \
      -s ${CHRONOSTRAIN_CLUSTER_FILE} \
      --aligner "bwa-mem2"

	  touch $filter_breadcrumb
  fi

  if [ -f ${inference_breadcrumb} ]; then
    echo "[*] Skipping inference for ${sample_name}"
  else
    echo "[*] Performing inference for ${sample_name}"
    export CHRONOSTRAIN_LOG_FILEPATH=${umb_plate_dir}/inference.log
    export CHRONOSTRAIN_CACHE_DIR=${umb_plate_dir}/chronostrain/cache
    mkdir -p ${umb_plate_dir}/chronostrain

    chronostrain advi \
      -r ${umb_plate_dir}/filtered/filtered_reads.csv \
      -o ${umb_plate_dir}/chronostrain \
      -s ${CHRONOSTRAIN_CLUSTER_FILE} \
      --correlation-mode $CHRONOSTRAIN_CORR_MODE \
      --iters $CHRONOSTRAIN_NUM_ITERS \
      --epochs $CHRONOSTRAIN_NUM_EPOCHS \
      --decay-lr $CHRONOSTRAIN_DECAY_LR \
      --lr-patience ${CHRONOSTRAIN_LR_PATIENCE} \
      --loss-tol ${CHRONOSTRAIN_LOSS_TOL} \
      --learning-rate ${CHRONOSTRAIN_LR} \
      --num-samples $CHRONOSTRAIN_NUM_SAMPLES \
      --read-batch-size $CHRONOSTRAIN_READ_BATCH_SZ \
      --min-lr ${CHRONOSTRAIN_MIN_LR} \
      --plot-format "pdf" \
      --plot-elbo \
      --prune-strains \
      --with-zeros \
      --prior-p 0.001 \
      --accumulate-gradients

    touch $inference_breadcrumb
  fi
done

#for umb_plate_dir in ${OUTPUT_DIR}/*
#do
#  sample_name="$(basename $umb_plate_dir)"
#  breadcrumb=${umb_plate_dir}/filter.DONE
#	export CHRONOSTRAIN_LOG_FILEPATH="${umb_plate_dir}/filter.log"
#
#  if [ -f $breadcrumb ]
#	then
#	  echo "Skipping filter for ${umb_id}."
#	else
#    echo "Filtering reads for ${sample_name}"
#    env JAX_PLATFORM_NAME=cpu chronostrain filter \
#      -r "${umb_plate_dir}/reads.csv" \
#      -o "${umb_plate_dir}/filtered" \
#      -s ${CHRONOSTRAIN_CLUSTER_FILE} \
#      --aligner "bwa-mem2"
#
#	  touch $breadcrumb
#	fi
#done
