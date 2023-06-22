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

  bash run_chronostrain.sh ${participant}
done