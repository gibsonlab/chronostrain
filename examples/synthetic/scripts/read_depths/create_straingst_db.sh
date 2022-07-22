#!/bin/bash
set -e
source settings.sh


# Step 1: Kmerize reference genomes.
mkdir -p ${STRAINGST_DB_DIR}
cd ${STRAINGST_DB_DIR}

cp ${CHRONOSTRAIN_DB_DIR}/assemblies/CP009273.1_Original/CP009273.1.fasta CP009273.1_Original.fasta
cp ${CHRONOSTRAIN_DB_DIR}/assemblies/CP009273.1_Substitution/CP009273.1_Substitution.fasta CP009273.1_Substitution.fasta
straingst kmerize -o CP009273.1_Original.hdf5 CP009273.1_Original.fasta
straingst kmerize -o CP009273.1_Substitution.hdf5 CP009273.1_Substitution.fasta
straingst kmersim --all-vs-all -t 4 --full-db -S jaccard -S subset *.hdf5 > similarities.tsv
straingst createdb -o ${STRAINGST_DB_HDF5} CP009273.1_Original.hdf5 CP009273.1_Substitution.hdf5
