#!/bin/bash
set -e
source settings.sh


mkdir -p ${STRAINGST_DB_DIR}
cd ${STRAINGST_DB_DIR}

python ${BASE_DIR}/helpers/list_strain_paths.py -j ${CHRONOSTRAIN_DB_JSON} -i $REFSEQ_INDEX \
| while read strain_seq; do
	base_name="$(basename -- $strain_seq)"
	echo "Kmerizing ${base_name}"
	strain_kmers="${base_name}.hdf5"
	straingst kmerize -o $strain_kmers $strain_seq
done

all_strain_kmers=""
for f in *.fna.hdf5; do
	all_strain_kmers="${all_strain_kmers} ${f}"
done
straingst createdb -o ${STRAINGST_DB_HDF5} ${all_strain_kmers}
