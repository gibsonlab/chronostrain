#!/bin/bash
set -e
source ../settings.sh
export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/create_clermontyping_input.log"

CLERMONTYPING_SCRIPT=~/ClermonTyping/clermonTyping.sh
echo "Using ClermonTyping script at: ${CLERMONTYPING_SCRIPT}"
echo "If ClermonTyping script does not exist, install it and/or change the path in this script (run_clermontyping.sh)."

python ${BASE_DIR}/scripts/phylogeny/create_clermontyping_input.py \
-i ${NCBI_REFSEQ_DIR}/index.tsv  \
-c ${CLERMONTYPING_SCRIPT} \
-o ${PHYLOGENY_OUTPUT_DIR}/ClermonTyping/clermontyping.sh \

bash ${PHYLOGENY_OUTPUT_DIR}/ClermonTyping/clermontyping.sh
