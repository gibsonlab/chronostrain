#!/bin/bash

# Database initialization. (pre-download fasta and markers.)

source settings.sh
CHRONOSTRAIN_LOG_FILEPATH=LOG_FILEPATH="${CHRONOSTRAIN_DATA_DIR}/logs/db_init/db_init.log"
export CHRONOSTRAIN_LOG_FILEPATH
python $BASE_DIR/scripts/helpers/initialize_chronostrain_database.py
