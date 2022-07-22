#!/bin/bash
set -e
source settings.sh

n_reads=$1
trial=$2
time_point=$3

if [ -z "$n_reads" ]
then
	echo "var \"n_reads\" is empty"
	exit 1
fi

if [ -z "$trial" ]
then
	echo "var \"trial\" is empty"
	exit 1
fi

trial_dir=$(get_trial_dir $n_reads $trial)
read_dir=${trial_dir}/reads
output_dir=${trial_dir}/output/strainphlan

echo "[*] Running StrainPhlAn for n_reads: ${n_reads}, trial: ${trial}"

samdir=${output_dir}/sams
bt2dir=${output_dir}/bowtie2
profiledir=${output_dir}/profiles
consensusdir=${output_dir}/consensus_markers
markerdir=${output_dir}/db_markers
mkdir -p ${output_dir}
mkdir -p ${samdir}
mkdir -p ${bt2dir}
mkdir -p ${profiledir}
mkdir -p ${consensusdir}
mkdir -p ${markerdir}


# Build bowtie2 db
python ${BASE_DIR}/helpers/build_metaphlan_db.py \
-o ${METAPHLAN3_DB_DIR}


# Run metaphlan
for time_point in 0 1 2 3 4; do
	read1="${read_dir}/${time_point}_reads_1.fq.gz"
	read2="${read_dir}/${time_point}_reads_2.fq.gz"
	metaphlan ${read1},${read2} \
	--input_type fastq \
	--index database \
	--bowtie2db ${METAPHLAN3_DB_DIR} \
	-s ${samdir}/${time_point}.sam.bz2 \
	--bowtie2out ${bt2dir}/${time_point}.bowtie2.bz2 \
	-o ${profiledir}/${time_point}_profiled.tsv
done

sample2markers.py -i ${samdir}/*.sam.bz2 --input_type sam -o ${consensusdir} -n 8
#extract_markers.py -c s__Escherichia_coli -o ${markerdir}
#strainphlan \
#-s ${consensusdir}/*.pkl \
#-m ${markerdir}/s__Escherichia_coli.fna \
#-r
