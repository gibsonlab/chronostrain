#!/bin/bash
export CHRONOSTRAIN_CACHE_DIR=/mnt/d/datasets/semisynthetic/trial_1/output/chronostrain/cache
export CHRONOSTRAIN_DB_DIR=/mnt/d/datasets/semisynthetic/databases/chronostrain
export CHRONOSTRAIN_DB_JSON=/mnt/d/datasets/semisynthetic/databases/chronostrain/all_strains.json
export CHRONOSTRAIN_INI=/home/younhun/chronostrain/examples/semisynthetic/files/chronostrain.ini
export CHRONOSTRAIN_LOG_INI=/home/younhun/chronostrain/examples/semisynthetic/files/logging.ini

chronostrain advi \
-r /mnt/d/datasets/semisynthetic/trial_1/output/chronostrain/filtered/filtered_input_files.csv \
-o /mnt/d/datasets/semisynthetic/trial_1/output/chronostrain \
--correlation-mode "full" \
--seed 31415 \
--iters 50 \
--epochs 1000 \
--decay-lr 0.25 \
--lr-patience 5 \
--min-lr 1e-5 \
--learning-rate 0.001 \
--num-samples 200 \
--read-batch-size 2500 \
--plot-format "pdf" \
--plot-elbo

