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

  filter_mark=${participant_dir}/chronostrain_99_99pct/filter.DONE
  if [[ -f ${filter_mark} ]]; then
    echo "[*] Skipping filter for participant ${participant}"
  else
    bash chronostrain/filter_chronostrain.99_99pct.sh ${participant}
    touch $filter_mark
  fi

  inference_mark=${participant_dir}/chronostrain_99_99pct/inference.DONE
  if [[ -f ${inference_mark} ]]; then
    echo "[*] Skipping inference for participant ${participant}"
  else
    bash chronostrain/run_chronostrain.99_99pct.sh ${participant}
    touch $inference_mark
  fi
done < infant_subset_tmp.txt
#done < infant_subset_99_99pct.txt
#done < "${INFANT_ID_LIST}"
