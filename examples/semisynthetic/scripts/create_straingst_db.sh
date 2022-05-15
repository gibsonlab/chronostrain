#!/bin/bash
set -e
source settings.sh


mkdir -p ${STRAINGST_DB_DIR}
cd ${STRAINGST_DB_DIR}

all_strain_kmers=""

python ${BASE_DIR}/helpers/strainest_helper.py \
-j ${CHRONOSTRAIN_DB_JSON} \
-i $REFSEQ_INDEX | while read strain_seq; do
	base_name="$(basename -- $strain_seq)"
	strain_kmers="${base_name}.hdf5"
	all_strain_kmers="${all_strain_kmers} ${strain_kmers}"
	straingst kmerize -o $strain_kmers $strain_seq
done

straingst createdb -o ${STRAINGST_DB_HDF5} ${all_strain_kmers}
