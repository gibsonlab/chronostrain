#!/bin/bash
set -e
source settings.sh
source msweep/settings.sh


env \
  CHRONOSTRAIN_LOG_FILEPATH=/mnt/e/semisynthetic_data/msweep_thresholds/reference_calculations.log \
  python msweep/detection_thresholds/reference_calculations.py
