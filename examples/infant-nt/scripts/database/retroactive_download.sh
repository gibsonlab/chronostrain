#!/bin/bash
set -e
source settings.sh

ref_genome_dir=${DATA_DIR}/ref_genomes
index_file=${ref_genome_dir}/index.tsv
mkdir -p "${ref_genome_dir}"

echo -e "Genus\tSpecies\tStrain\tAccession\tAssembly\tSeqPath\tChromosomeLen\tGFF" > ${index_file}
python database/list_ncbi_accessions.py -c ${CHRONOSTRAIN_DB_JSON} | while IFS=$',' read -r strain_id strain_name assembly_acc genus species chrom_len; do
  echo "[!] Handling ${assembly_acc}"
    
#   echo "${strain_id}|${strain_name}|${assembly_acc}|${genus}|${species}|${chrom_len}"
#   continue

  target_dir=${ref_genome_dir}/${assembly_acc}
  chrom_file="${target_dir}/${strain_id}.chrom.fna"
  chrom_file_gz="${chrom_file}.gz"
  gff_file=${target_dir}/genomic.gff
  gff_file_gz="${gff_file}.gz"

  if [ -f ${chrom_file}.gz ]; then
    echo "[!] ${assembly_acc} already downloaded at ${chrom_file}"
  else
    echo "[!] Downloading ${assembly_acc}"
    
    # Download and extract.
    cd "${target_dir}"
    datasets download genome accession "${assembly_acc}" --include genome,gff3
    unzip -j ncbi_dataset.zip
    
    # Clean up.
    rm *.json
    rm *.jsonl
    rm README.md
    rm ncbi_dataset.zip
    cd -
    
    # Extract chromosomal fasta file.
    fasta_file=$(find ${target_dir} | grep genomic.fna)
    python database/extract_chromosome.py -i ${fasta_file} -o ${chrom_file}
    
    # compress everything.
    pigz ${fasta_file}
    pigz ${chrom_file}
    if [ -f ${gff_file} ]; then pigz ${gff_file}; fi
  fi
  
  if [ -f ${gff_file_gz} ]; then
    echo -e "${genus}\t${species}\t${strain_name}\t${strain_id}\t${assembly_acc}\t${chrom_file_gz}\t${chrom_len}\t${gff_file_gz}" >> ${index_file}
  else
    echo -e "${genus}\t${species}\t${strain_name}\t${strain_id}\t${assembly_acc}\t${chrom_file_gz}\t${chrom_len}\t" >> ${index_file}
  fi
done

echo Compiled index at: ${index_file}
