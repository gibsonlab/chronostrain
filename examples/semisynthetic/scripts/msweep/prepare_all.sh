#!/bin/bash
set -e
source settings.sh

cd ${BASE_DIR}/scripts
bash msweep/prepare_pseudoalignment_index.sh
