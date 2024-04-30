#!/bin/bash
set -e
source settings.sh
source msweep/settings.sh

require_program themisto

# ============ Requires arguments:
participant=$1
require_variable 'participant' $participant


# ============ script body:
participant_dir=${DATA_DIR}/${participant}
output_dir=${participant_dir}/themisto
breadcrumb=${output_dir}/themisto.DONE

if [ -f ${breadcrumb} ]; then
  echo "[*] Pseudoalignment for ${participant} already done."
  exit 0
fi


echo "[**] Preparing themisto inputs for ${participant}..."
mkdir -p ${output_dir}
input_file=${output_dir}/query_files.txt
output_file=${output_dir}/output_files.txt
> $input_file
> $output_file


index_name=${THEMISTO_DB_DIR}/${THEMISTO_DB_NAME}
tmp_dir="${output_dir}/_tmp"
n_ref=$(wc -l < "${THEMISTO_DB_DIR}/clusters.txt")

while IFS=$'\t' read part_id time_point sample_id read1_raw_fq read2_raw_fq
do
    if [ "${part_id}" == "Participant" ]; then continue; fi
    echo "[**] Running pseudoalignment ${participant}: ${sample_id} (timepoint ${time_point})"

    fq_1="${participant_dir}/reads/${sample_id}_kneaddata/${sample_id}_paired_1.fastq.gz"
    fq_2="${participant_dir}/reads/${sample_id}_kneaddata/${sample_id}_paired_2.fastq.gz"
    tmp_1="${output_dir}/${sample_id}_paired_1.out.txt"
    tmp_2="${output_dir}/${sample_id}_paired_2.out.txt"
    out_1="${output_dir}/${sample_id}_paired_1.aln"
    out_2="${output_dir}/${sample_id}_paired_2.aln"

    themisto pseudoalign -q $fq_1 -i $index_name --temp-dir $tmp_dir --n-threads $N_CORES --sort-output -o ${tmp_1}
    themisto pseudoalign -q $fq_2 -i $index_name --temp-dir $tmp_dir --n-threads $N_CORES --sort-output -o ${tmp_2}

    n_reads_1=$(wc -l < "${tmp_1}")
    n_reads_2=$(wc -l < "${tmp_2}")
    if [[ $n_reads_1 != $n_reads_2 ]]; then
	    echo "# of reads in pseudoalignments don't match (${n_reads_1} vs ${n_reads_2})."
	    exit 1
    fi
    
    alignment-writer -n $n_ref -r $n_reads_1 -f $tmp_1 > $out_1
    alignment-writer -n $n_ref -r $n_reads_2 -f $tmp_2 > $out_2
    rm $tmp_1
    rm $tmp_2
done < ${participant_dir}/dataset.tsv


# clean up.
echo '[**] Cleaning up...'
rm -rf $tmp_dir
touch ${breadcrumb}
