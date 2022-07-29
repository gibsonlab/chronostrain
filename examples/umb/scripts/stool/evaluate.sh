#!/bin/bash
set -e

source settings.sh
SEED=31415



# =========== Run chronostrain. ==================
echo "Running ChronoStrain evaluation. (ungrouped)"
python ${BASE_DIR}/helpers/eval_chronostrain.py -d $CHRONOSTRAIN_OUTPUT_DIR -o $OUTPUT_DIR/chronostrain.csv


echo "Running ChronoStrain evaluation. (grouped)"
python ${BASE_DIR}/helpers/eval_chronostrain.py -d $CHRONOSTRAIN_OUTPUT_DIR -o $OUTPUT_DIR/chronostrain.grouped.csv \
--group_by_clades -c $CLADES_FILE


echo "Running StrainGE evaluation. (ungrouped)"
python ${BASE_DIR}/helpers/eval_strainge.py -d $STRAINGE_OUTPUT_DIR -o $OUTPUT_DIR/strainge.csv -m $SRA_CSV_PATH \
-r ${REFSEQ_INDEX}


echo "Running StrainGE evaluation. (grouped)"
python ${BASE_DIR}/helpers/eval_strainge.py -d $STRAINGE_OUTPUT_DIR -o $OUTPUT_DIR/strainge.grouped.csv -m $SRA_CSV_PATH \
-r ${REFSEQ_INDEX} \
--group_by_clades -c $CLADES_FILE
