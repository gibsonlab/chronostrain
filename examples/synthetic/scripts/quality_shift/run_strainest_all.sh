#!/bin/bash
set -e
source settings.sh


mkdir -p ${STRAINEST_DB_DIR}
cd ${STRAINEST_DB_DIR}
cp ${CHRONOSTRAIN_DB_DIR}/assemblies/CP009273.1_Original/CP009273.1_Original.fasta CP009273.1_Original.fasta
cp ${CHRONOSTRAIN_DB_DIR}/assemblies/CP009273.1_Substitution/CP009273.1_Substitution.fasta CP009273.1_Substitution.fasta

echo "[*] Building bowtie2 index..."
bowtie2-build CP009273.1_Original.fasta,CP009273.1_Substitution.fasta -f ${STRAINEST_BOWTIE2_DB_NAME} --quiet


cd ${BASE_DIR}/scripts/quality_shift
for (( q_shift = ${Q_SHIFT_MIN}; q_shift < ${Q_SHIFT_MAX}+1; q_shift += ${Q_SHIFT_STEP} ));
do
	for (( trial = 1; trial < ${N_TRIALS}+1; trial++ ));
	do
		bash run_strainest.sh $q_shift $trial 0
		bash run_strainest.sh $q_shift $trial 1
		bash run_strainest.sh $q_shift $trial 2
		bash run_strainest.sh $q_shift $trial 3
		bash run_strainest.sh $q_shift $trial 4
	done
done