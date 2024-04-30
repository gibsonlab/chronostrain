#!/bin/bash
set -e

source settings.sh

# =========== Run chronostrain. ==================
echo "Note: this script assumes that the database JSON files were constructed using the Jupyter Notebook in the \"examples\" directory. All three versions of the JSON files are required."


env JAX_PLATFORM_NAME=cpu \
  chronostrain cluster-json \
  -i /mnt/e/ecoli_db/chronostrain_files/ecoli.json \
  -o /mnt/e/ecoli_db/chronostrain_files/ecoli.99_95pct.txt \
  --ident-threshold 0.9995



umb_id="UMB18"
run_dir=${OUTPUT_DIR}/${umb_id}
breadcrumb=${run_dir}/inference.DONE
filter_breadcrumb=${run_dir}/filter.DONE
granular_breadcrumb=${run_dir}/inference_99_95pct.DONE

if ! [ -f $filter_breadcrumb ]; then
  echo "[*] Filter not done for ${umb_id}."
elif ! [ -f $breadcrumb ]; then
  echo "[*] Regular inference not done for ${umb_id}."
else
  echo "[*] Running granular inference for ${umb_id}"
  export CHRONOSTRAIN_LOG_FILEPATH=${run_dir}/logs/chronostrain_inference.log
  export CHRONOSTRAIN_CACHE_DIR=${run_dir}/chronostrain/cache
  mkdir -p ${run_dir}/chronostrain

  #echo "[**] Extracting from previous run..."
  #python ${BASE_DIR}/helpers/granular_extract.py \
  #  -r ${run_dir}/filtered/filtered_reads.csv \
  #  -c ${run_dir}/chronostrain \
  #  --coarse-clustering /mnt/e/ecoli_db/chronostrain_files/ecoli.clusters.txt \
  #  -g /mnt/e/ecoli_db/chronostrain_files/ecoli.medium_clustering.txt \
  #  -o ${run_dir}/chronostrain_medium/target_ids.txt \
  #  --prior-p 0.001 \
  #  --abund-lb 0.05 \
  #  --bayes-factor 100000.0

  echo "[**] Running new inference..."
  #    -s ${run_dir}/chronostrain_medium/target_ids.txt \
  #    --prior-p 0.5
  chronostrain advi \
    -r ${run_dir}/filtered/filtered_reads.csv \
    -o ${run_dir}/chronostrain_99_95pct \
    -s /mnt/e/ecoli_db/chronostrain_files/ecoli.99_95pct.txt \
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

  touch $granular_breadcrumb
fi
# ================================================
