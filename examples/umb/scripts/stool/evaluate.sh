#!/bin/bash
set -e

source settings.sh
SEED=31415


# =========== Run chronostrain. ==================
echo "Running ChronoStrain evaluation. (ungrouped)"
python ${BASE_DIR}/helpers/eval_chronostrain.py \
-d $CHRONOSTRAIN_OUTPUT_DIR \
-o $OUTPUT_DIR/chronostrain.tsv \
-r ${READS_DIR} \
-lb 0.0014409


echo "Running ChronoStrain evaluation. (grouped)"
python ${BASE_DIR}/helpers/eval_chronostrain.py \
-d $CHRONOSTRAIN_OUTPUT_DIR \
-o $OUTPUT_DIR/chronostrain.grouped.tsv \
-r ${READS_DIR} \
-lb 0.0014409 \
--group_by_clades -c $CLADES_FILE


echo "Running StrainGE evaluation. (ungrouped)"
python ${BASE_DIR}/helpers/eval_strainge.py \
-d $STRAINGE_OUTPUT_DIR \
-o $OUTPUT_DIR/strainge.tsv -m $SRA_CSV_PATH \
-r ${REFSEQ_INDEX} \
-m $SRA_CSV_PATH


echo "Running StrainGE evaluation. (grouped)"
python ${BASE_DIR}/helpers/eval_strainge.py \
-d $STRAINGE_OUTPUT_DIR \
-o $OUTPUT_DIR/strainge.grouped.tsv \
-r ${REFSEQ_INDEX} \
-m $SRA_CSV_PATH \
--group_by_clades -c $CLADES_FILE
