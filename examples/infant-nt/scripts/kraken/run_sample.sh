#!/bin/bash
source settings.sh
source kraken/settings.sh
set -e


infant_id=$1
sample_id=$2

# run kraken on sample
fq1_gz=${DATA_DIR}/${infant_id}/reads/${sample_id}_1.fastq.gz
fq2_gz=${DATA_DIR}/${infant_id}/reads/${sample_id}_2.fastq.gz

out_dir=${DATA_DIR}/${infant_id}/kraken/${sample_id}
kraken_report=${out_dir}/kraken.report
kraken_out=${out_dir}/kraken.out
bracken_out=${out_dir}/bracken.out
breadcrumb=${out_dir}/bracken.DONE

echo "breadcrumb=${breadcrumb}"
if [ -f ${breadcrumb} ]; then
  echo "[*] Kraken2 + Bracken already done for ${infant_id}, sample ${sample_id}"
fi

mkdir -p ${out_dir}
echo "[*] Running kraken2 + bracken for ${infant_id}, sample ${sample_id}"
echo "Using DBs from ${KRAKEN2_DB_PATH}"
kraken2 --paired --gzip-compressed --db ${KRAKEN2_DB_NAME} --threads ${N_THREADS} --report ${kraken_report} ${fq1_gz} ${fq2_gz} > ${kraken_out}
bracken -d ${KRAKEN2_DB_PATH}/${KRAKEN2_DB_NAME} -i ${kraken_report} -o ${bracken_out} -r ${KMER_LEN} -l S
touch ${breadcrumb}
