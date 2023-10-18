#!/bin/bash
set -e

source settings.sh

# =========== Run chronostrain. ==================
for umb_plate_dir in ${OUTPUT_DIR}/*
do
  sample_name="$(basename $umb_plate_dir)"
  breadcrumb=${umb_plate_dir}/inference.DONE
  filter_breadcrumb=${umb_plate_dir}/filter.DONE

  if [ -f $breadcrumb ]; then
    echo "[*] Skipping inference for ${sample_name}."
  elif ! [ -f $filter_breadcrumb ]; then
    echo "[*] Filter not done for ${sample_name}."
  else
    echo "[*] Running inference for ${sample_name}"
    export CHRONOSTRAIN_LOG_FILEPATH=${umb_plate_dir}/inference.log
    export CHRONOSTRAIN_CACHE_DIR=${umb_plate_dir}/chronostrain/cache
    mkdir -p ${umb_plate_dir}/chronostrain

    chronostrain advi \
      -r ${umb_plate_dir}/filtered/filtered_reads.csv \
      -o ${umb_plate_dir}/chronostrain \
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

    touch $breadcrumb
  fi
done
# ================================================