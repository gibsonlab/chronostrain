#!/bin/bash
set -e
source settings.sh

# Run FastMLST on RefSeq database.

fastmlst_outdir="${CHRONOSTRAIN_DB_DIR}/fastmlst"
fastmlst_genome_tmp="${fastmlst_outdir}/_tmp"
st_output="${fastmlst_outdir}/fastmlst.tsv"
echo "Output of FastMLST will be written to: ${st_output}"


mkdir -p "${fastmlst_genome_tmp}"

echo "[*] Linking RefSeq catalog..."
while IFS=$'\t' read -r -a row; do
  acc="${row[3]}"
  species="${row[1]}"
  seq_path="${row[5]}"
  if [ "${acc}" == "Accession" ]; then continue; fi
  if [ "${species}" != "coli" ]; then continue; fi
  ln -s "${seq_path}" "${fastmlst_genome_tmp}/${acc}"
done < "${REFSEQ_INDEX}"


> "${st_output}"
fastmlst -s '\t' ${fastmlst_genome_tmp}/* > ${st_output}
rm -rf "${fastmlst_genome_tmp}"
