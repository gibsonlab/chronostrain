#!/bin/bash
set -e
source ../settings.sh
export CHRONOSTRAIN_LOG_FILEPATH="${LOGDIR}/create_clermontyping_input.log"

CLERMONTYPING_SCRIPT=~/ClermonTyping/clermonTyping.sh
echo "[*] Using ClermonTyping script at: ${CLERMONTYPING_SCRIPT}"
echo "[*] If ClermonTyping script is not found, install it and/or change the path in this script (run_clermontyping.sh)."

python ${BASE_DIR}/scripts/phylogeny/create_clermontyping_input.py \
-i ${NCBI_REFSEQ_DIR}/index.tsv  \
-c ${CLERMONTYPING_SCRIPT} \
-o ${PHYLOGENY_OUTPUT_DIR}/ClermonTyping/clermontyping.sh \

echo "[*] Running ClermonTyping batched analysis."
cd ${PHYLOGENY_OUTPUT_DIR}/ClermonTyping
bash clermontyping.sh

# Merge all batched results.
final_path=${PHYLOGENY_OUTPUT_DIR}/ClermonTyping/umb_phylogroups_complete.txt
> $final_path  # Clear contents of file.
for f in ${PHYLOGENY_OUTPUT_DIR}/ClermonTyping/umb_*/umb_*_phylogroups.txt
do
	cat $f >> $final_path
done
echo "[*] Generated phylogroup path ${final_pathk}."
