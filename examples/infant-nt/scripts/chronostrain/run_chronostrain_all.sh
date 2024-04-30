#!/bin/bash
set -e
source settings.sh
source chronostrain/settings.sh


while read line
do
  participant=$line
  echo $participant
  participant_dir=${DATA_DIR}/${participant}

  if [[ ! -d $participant_dir ]]; then
    echo "${participant_dir} doesn't exist!"
    continue
  fi

  filter_mark=${participant_dir}/chronostrain/filter.DONE
  if [[ -f ${filter_mark} ]]; then
    echo "[*] Skipping filter for participant ${participant}"
  else
    bash chronostrain/filter_chronostrain.sh ${participant}
    touch $filter_mark
  fi

  inference_mark=${participant_dir}/chronostrain/inference.DONE
  if [[ -f ${inference_mark} ]]; then
    echo ${inference_mark}
    echo "[*] Skipping inference for participant ${participant}"
  else
    bash chronostrain/run_chronostrain.sh ${participant}
    touch $inference_mark
  fi
done < infant_subset_tmp.txt
#done < "${INFANT_ID_LIST}"
