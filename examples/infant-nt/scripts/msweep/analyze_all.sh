#!/bin/bash
set -e
source settings.sh


while read line
do
  participant=$line
  if [ "${participant}" == "A00502" ] || [ "${participant}" == "A01687" ] || [ "${participant}" == "A01966" ]; then
      echo "[!!!] Skipping ${participant}"
      continue
  fi

  bash msweep/run_pseudoalignment.sh ${participant}
  bash msweep/run_msweep.sh ${participant}
done < "${INFANT_ID_LIST}"
