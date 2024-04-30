#!/bin/bash
set -e
source settings.sh
source msweep/settings.sh

require_program mSWEEP

# ============ Requires arguments:
participant=$1
require_variable 'participant' $participant

participant_dir=${DATA_DIR}/${participant}
output_dir=${participant_dir}/msweep
breadcrumb=${output_dir}/msweep.DONE
if [ -f ${breadcrumb} ]; then
  echo "[*] mSWEEP inference for ${participant} already done."
  exit 0
fi

pseudoalign_dir=${participant_dir}/themisto
pseudoalignment_breadcrumb=${pseudoalign_dir}/themisto.DONE
if ! [ -f ${pseudoalignment_breadcrumb} ]; then
  echo "[*] Pseudoalignments for ${participant} not yet done."
  exit 1
fi


n_ref=$(wc -l < "${THEMISTO_DB_DIR}/clusters.txt")
echo "[**] Will use n_ref=${n_ref} for alignment compression."

mkdir -p ${output_dir}
while IFS=$'\t' read part_id time_point sample_id read1_raw_fq read2_raw_fq
do
    if [ "${part_id}" == "Participant" ]; then continue; fi

    fwd_input=${pseudoalign_dir}/${sample_id}_paired_1.aln
    rev_input=${pseudoalign_dir}/${sample_id}_paired_2.aln
    if ! [ -f ${fwd_input} ]; then echo "Forward read pseudoalignment not found."; exit 1; fi
    if ! [ -f ${rev_input} ]; then echo "Reverse read pseudoalignment not found."; exit 1; fi

    echo "[*] Running mSWEEP ${sample_id} (timepoint ${time_point})"
    cd ${output_dir}
    echo "USING CLUSTERS FROM ${THEMISTO_DB_DIR}"
    mSWEEP \
    --themisto-1 ${fwd_input} \
	  --themisto-2 ${rev_input} \
	  -i ${THEMISTO_DB_DIR}/clusters.txt \
	  -t ${N_CORES} \
	  --bin-reads \
	  --min-abundance 0.01 \
	  -o ${sample_id}
    cd -
done < ${participant_dir}/dataset.tsv
touch ${breadcrumb}
