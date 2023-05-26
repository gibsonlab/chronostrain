#!/bin/bash
set -e
source settings.sh


for f in /mnt/e/infant_nt/*; do
  if [[ ! -d $f ]]; then
    continue
  fi

  if [[ ! -f "${f}/chronostrain/process_reads.DONE" ]]; then
    continue
  fi

  participant="$(basename ${f})"
  chronostrain_analysis_mark=${f}/chronostrain/analysis.DONE
  if [[ -f ${chronostrain_analysis_mark} ]]; then
    echo "[*] Skipping analysis for participant ${participant}"
    continue
  fi

  echo "[*] Running analysis for participant ${participant}"
  bash run_chronostrain.sh ${participant}
  touch $chronostrain_analysis_mark
done