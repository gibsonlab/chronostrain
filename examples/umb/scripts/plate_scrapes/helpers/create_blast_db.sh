#!/bin/bash
set -e

source settings.sh


cd ${CHRONOSTRAIN_MARKERS_DIR}

echo "[*] Making blast DB."
makeblastdb \
-in ${MARKER_FASTA} \
-out ${BLAST_DB_NAME} \
-dbtype nucl \
-title "ChronoStrain Markers" \
-parse_seqids
