#!/bin/bash
source settings.sh
source metaphlan/settings.sh
set -e


infant_id=$1
sample_id=$2

# run metaphlan on sample
fq1_gz=${DATA_DIR}/${infant_id}/reads/${sample_id}_1.fastq.gz
fq2_gz=${DATA_DIR}/${infant_id}/reads/${sample_id}_2.fastq.gz

out_dir=${DATA_DIR}/${infant_id}/metaphlan/${sample_id}
bt2_out=${out_dir}/bowtie2.bz2
profile_out=${out_dir}/profiled_metagenome.txt
breadcrumb=${out_dir}/metaphlan.DONE

echo "breadcrumb=${breadcrumb}"
if [ -f ${breadcrumb} ]; then
  echo "[*] MetaPhlAn already done for ${infant_id}, sample ${sample_id}"
  exit 0
fi

mkdir -p ${out_dir}
echo "[*] Running MetaPhlAn for ${infant_id}, sample ${sample_id}"
metaphlan ${fq1_gz},${fq2_gz} \
  --nproc ${METAPHLAN_N_THREADS} \
  --bowtie2out ${bt2_out} \
  --input_type fastq \
  -o ${profile_out} \
  --bowtie2db ${METAPHLAN_DB_DIR} \
  --index "${METAPHLAN_DB_NAME}"

rm ${bt2_out}
touch ${breadcrumb}
