#!/bin/bash
source settings.sh


python ${BASE_DIR}/helpers/list_all_participants.py ${ENA_ISOLATE_ASSEMBLY_CATALOG} | while read line
do
  participant=$line
  echo $participant
  participant_dir=${DATA_DIR}/${participant}

  if [[ ! -d $participant_dir ]]; then
    continue
  fi

  inference_mark=${participant_dir}/chronostrain/inference.DONE
  if [[ -f ${inference_mark} ]]; then
    bash quantify_chronostrain.sh $participant
  fi
done