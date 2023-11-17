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
  seq_path="${row[5]}"
  if [ "${acc}" == "Accession" ]; then continue; fi
  ln -s "${seq_path}" "${fastmlst_genome_tmp}/${acc}"
done < "${REFSEQ_INDEX}"


echo "[*] Linking infant isolate catalog..."
for metadata_file in ${DATA_DIR}/*/isolate_assemblies/metadata.tsv; do
  while IFS=$'\t' read -r -a metadata_row; do
    acc="${metadata_row[1]}"
    seq_path="${metadata_row[2]}"
    if [ "${acc}" == "Accession" ]; then continue; fi
    ln -s "${seq_path}" "${fastmlst_genome_tmp}/${acc}"
  done < "${metadata_file}"
done


> "${st_output}"
fastmlst -s '\t' ${fastmlst_genome_tmp}/* > ${st_output}
rm -rf "${fastmlst_genome_tmp}"
