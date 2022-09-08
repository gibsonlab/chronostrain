#!/bin/bash
set -e
source settings.sh


# ================= Database using complete genomes
mkdir -p ${STRAINGST_DB_DIR}/full_genome_hdf5
cd ${STRAINGST_DB_DIR}/full_genome_hdf5

references_file=$STRAINGST_DB_DIR/full_genome_hdf5/references_list.txt
> $references_file

python ${BASE_DIR}/helpers/list_strain_paths.py -j ${CHRONOSTRAIN_DB_JSON} -i $REFSEQ_INDEX \
| while read strain_seq; do
	base_name="$(basename -- $strain_seq)"
	base_name="${base_name%.chrom.fna}"
	echo "Kmerizing ${base_name} chromosomes..."
	strain_kmers="${base_name}.hdf5"
	straingst kmerize -o $strain_kmers $strain_seq

	echo "$strain_kmers" >> $references_file
done

all_strain_kmers=""
for f in *.hdf5; do
	all_strain_kmers="${all_strain_kmers} ${f}"
done
straingst createdb -o ${STRAINGST_CHROMOSOME_DB_HDF5} -f $references_file
rm -rf ${STRAINGST_DB_DIR}/full_genome_hdf5


## ================= Database using marker sequences
#mkdir -p ${STRAINGST_DB_DIR}/markers
#cd ${STRAINGST_DB_DIR}/markers
#
#python ${BASE_DIR}/helpers/chronostrain_markers_to_fasta.py -o .
#for f in *.markers.fasta; do
#	base_name="${f%.markers.fasta}"
#	echo "Kmerizing $base_name markers..."
#	strain_kmers="${base_name}.hdf5"
#	straingst kmerize -o $strain_kmers $f
#done
#
#all_strain_kmers=""
#for f in *.hdf5; do
#	all_strain_kmers="${all_strain_kmers} ${f}"
#done
#straingst createdb -o ${STRAINGST_MARKER_DB_HDF5} ${all_strain_kmers}
