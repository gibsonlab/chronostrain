#!/bin/bash
set -e

source settings.sh

# =========== Run chronostrain. ==================
echo "Note: this script assumes that the database JSON files were constructed using the Jupyter Notebook in the `examples` directory. All three versions of the JSON files are required."


# First, re-cluster the original database.
env JAX_PLATFORM_NAME=cpu \
  chronostrain prune-json \
  -i asdf \
  -o asdf \
  --ident-threshold 0.9999999999



umb_id="UMB18"
echo "[*] Running granular inference on ${umb_id}."
run_dir=${OUTPUT_DIR}/${umb_id}
breadcrumb=${run_dir}/inference.DONE
filter_breadcrumb=${run_dir}/filter.DONE
granular_breadcrumb=${run_dir}/inference_granular.DONE

if [ -f $granular_breadcrumb ]; then
  echo "[*] Skipping granular inference for ${umb_id}."
elif ! [ -f $filter_breadcrumb ]; then
  echo "[*] Filter not done for ${umb_id}."
elif ! [ -f $breadcrumb ]; then
  echo "[*] Regular inference not done for ${umb_id}."
else
  echo "[*] Running granular inference for ${umb_id}"
  export CHRONOSTRAIN_LOG_FILEPATH=${run_dir}/logs/chronostrain_inference.log
  export CHRONOSTRAIN_CACHE_DIR=${run_dir}/chronostrain/cache
  mkdir -p ${run_dir}/chronostrain

  python ${BASE_DIR}/helpers/granular_inference.py \
    -r ${run_dir}/filtered/filtered_reads.csv \
    -c ${run_dir}/chronostrain \
    -o ${run_dir}/chronostrain_granular \
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
# ================================================