#!/bin/bash
set -e
source settings.sh


python ${BASE_DIR}/helpers/list_all_participants.py ${NT_RESULT_TABLE} | while read line
do
  participant=$line
  echo "[*] Handling ${participant}"
  if [[ "${participant}" == "514124" ]]; then
    echo "Skipping. (TODO remove this check)"
  fi
  bash download_dataset.sh ${participant}
done
