#!/bin/bash

SLEEP_SECONDS=1800
while true; do
    echo "Retrying..."
    bash run_chronostrain_all.sh
    echo "Done with loop. Sleeping... (will retry after ${SLEEP_SECONDS} seconds.)"
    sleep ${SLEEP_SECONDS}
done
