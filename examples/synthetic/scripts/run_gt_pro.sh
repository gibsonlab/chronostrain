#!/bin/bash
set -e
source settings.sh


db_dir=${CHRONOSTRAIN_DB_DIR}/gt-pro
mkdir -p ${db_dir}
cd ${db_dir}

# strain genomes
mkdir genomes
cp ${CHRONOSTRAIN_DB_DIR}/assemblies/CP009273.1_Original/CP009273.1.fasta genomes/CP009273.1_Original.fasta
cp ${CHRONOSTRAIN_DB_DIR}/assemblies/CP009273.1_Substitution/CP009273.1_Substitution.fasta genomes/CP009273.1_Substitution.fasta

# reference
cp ${CHRONOSTRAIN_DB_DIR}/assemblies/CP009273.1_Original/CP009273.1.fasta reference.fna

# multiple alignment
> reference.fna
cat genomes/CP009273.1_Original.fasta >> msa.fa
cat genomes/CP009273.1_Substitution.fasta >> msa.fa
