#!/bin/bash
set -e
source settings.sh


python ${BASE_DIR}/helpers/list_all_participants.py ${ENA_ISOLATE_ASSEMBLY_CATALOG} | while read line
do
  participant=$line
  echo $participant
  participant_dir=${DATA_DIR}/${participant}

  if [[ ! -d $participant_dir ]]; then
    continue
  fi

  if [[ ! -f "${participant_dir}/chronostrain/process_reads.DONE" ]]; then
    continue
  fi

  filter_mark=${participant_dir}/chronostrain/filter.DONE
  if [[ -f ${filter_mark} ]]; then
    echo "[*] Skipping filter for participant ${participant}"
  else
    bash filter_chronostrain.sh ${participant}
    touch $filter_mark
  fi

  inference_mark=${participant_dir}/chronostrain/inference.DONE
  if [[ -f ${inference_mark} ]]; then
    echo "[*] Skipping inference for participant ${participant}"
  else
    bash run_chronostrain.sh ${participant}
    touch $inference_mark
  fi
done