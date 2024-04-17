#!/bin/bash
set -e
source settings.sh

# Run FastMLST on RefSeq database.

mash_outdir="${CHRONOSTRAIN_DB_DIR}/mash"
mash_genome_tmp="${mash_outdir}/_tmp"
mash_dist_file="${mash_outdir}/distances.txt"


mkdir -p "${mash_genome_tmp}"

echo "[*] Linking RefSeq catalog..."
while IFS=$'\t' read -r -a row; do
  acc="${row[3]}"
  species="${row[1]}"
  seq_path="${row[5]}"
  if [ "${acc}" == "Accession" ]; then continue; fi
  if [ "${species}" != "coli" ]; then continue; fi
  ln -s "${seq_path}" "${mash_genome_tmp}/${acc}"
done < "${REFSEQ_INDEX}"


# sketch
cd ${mash_genome_tmp}
mash sketch -o reference *

# clean up after sketch
mv reference.msh ../
cd ..
rm -rf ./_tmp

# compute dists
mash info reference.msh
mash triangle reference.msh > ${mash_dist_file} -p 8
