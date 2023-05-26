#!/bin/bash
set -e
source settings.sh


export CHRONOSTRAIN_LOG_FILEPATH="${OUTPUT_DIR}/logs/mlst_prune.log"
export CHRONOSTRAIN_DB_NAME="mlst"
chronostrain prune-pickle -i "mlst" -o "mlst_pruned" -t 0.001
