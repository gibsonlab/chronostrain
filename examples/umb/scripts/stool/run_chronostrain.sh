#!/bin/bash
set -e

source settings.sh

# =========== Run chronostrain. ==================
echo "Running inference."

for umb_id in UMB01 UMB02 UMB03 UMB04 UMB05 UMB06 UMB07 UMB08 UMB09 UMB10 UMB11 UMB12 UMB13 UMB14 UMB15 UMB16 UMB17 UMB18 UMB19 UMB20 UMB21 UMB22 UMB23 UMB24 UMB25 UMB26 UMB27 UMB28 UMB29 UMB30 UMB31
do
  echo "[*] Running inference on ${umb_id}."
  run_dir=${OUTPUT_DIR}/${umb_id}
  breadcrumb=${run_dir}/inference.DONE
  filter_breadcrumb=${run_dir}/filter.DONE

  if [ -f $breadcrumb ]; then
    echo "Skipping inference for ${umb_id}."
  elif ! [ -f $filter_breadcrumb ]; then
    echo "Filter not done for ${umb_id}."
  else
    export CHRONOSTRAIN_LOG_FILEPATH=${run_dir}/logs/chronostrain_inference.log
    export CHRONOSTRAIN_CACHE_DIR=${run_dir}/chronostrain/cache
    mkdir -p ${run_dir}/chronostrain

    chronostrain advi \
    -r ${run_dir}/filtered/filtered_reads.csv \
    -o ${run_dir}/chronostrain \
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