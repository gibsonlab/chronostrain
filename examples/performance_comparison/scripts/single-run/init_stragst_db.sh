source settings.sh

for genome_file in ${CHRONOSTRAIN_DB_DIR}/*.fasta
do
	bn=$(basename ${genome_file%.fasta})
	straingst kmerize \
		-o ${STRAINGE_DB_DIR}/${bn}.hdf5 \
		$genome_file
done
