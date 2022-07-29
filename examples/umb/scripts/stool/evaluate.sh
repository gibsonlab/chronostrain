#!/bin/bash
set -e

source settings.sh
SEED=31415

# =========== Run chronostrain. ==================
echo "Running evaluation."

python ${BASE_DIR}/helpers/eval_chronostrain.py -d $CHRONOSTRAIN_OUTPUT_DIR -o $OUTPUT_DIR/chronostrain.csv
python ${BASE_DIR}/helpers/eval_strainge.py -d $STRAINGE_OUTPUT_DIR -o $OUTPUT_DIR/strainge.csv -m $SRA_CSV_PATH
