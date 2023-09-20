#!/bin/bash
set -e
source settings.sh

export NUM_CORES=8

export TARGET_TAXA="Enterococcaceae"
export TARGET_DIR=/mnt/e/infant_nt/database  # all intermediate and final DB files will go here.

export NCBI_REFSEQ_DIR=${TARGET_DIR}/ref_genomes  # Directory to place the refseq assembly seqs.
export REFSEQ_INDEX=${TARGET_DIR}/ref_genomes/index.tsv  # The TSV index of the downloaded NCBI refseq assemblies.
export BLAST_DB_DIR=${TARGET_DIR}/blast_db  # where to store BLAST database.
export MARKER_SEED_INDEX=${TARGET_DIR}/marker_seeds/marker_seed_index.tsv  # place to store marker FASTA files
export CHRONOSTRAIN_TARGET_JSON=${TARGET_DIR}/efaecalis.json  # The final product.

export BLAST_DB_NAME=Efaecalis_refseq  # target BLAST database name.
export MIN_PCT_IDTY=75  # 90: "Markers" are given by 90% identity alignments.
export CHRONOSTRAIN_DB_DIR=${TARGET_DIR}/chronostrain_files


# ========= Main body
# For an explanation, refer to the README.

perform_primer_search()
{
  gene_name=$1
  primer1=$2
  primer2=$3
  amp_len=$4

  python python_helpers/identify_gene_cluster_primersearch.py \
    -i "${REFSEQ_INDEX}" \
    -t ${TARGET_DIR}/_tmp \
    -o "${TARGET_DIR}/marker_seeds/${gene_name}.feather" \
    -p1 "${primer1}" \
    -p2 "${primer2}" \
    -n "gene_${gene_name}" \
    -l
}


#==== Pathogenicity island
 #gls24-like *
 #cylB *
 #esp *
 #xylS homolog *
 #psaA homolog *
 #
 #
 #==== Virulence suspects
 #cylA (cytolysin)
 #cylB
 #cylM
 #cbh (bile salt hydrolase)
 #gelE (gelatinase)
 #fsrB (gelatinase regulator)
 #
 #* REMINDER ==========> include expected amplicon sizes!!!! (into command line interface)

bash download_ncbi2.sh
bash create_blast_db.sh
python python_helpers/mlst_download.py -t "Enterococcus faecalis" -w ${TARGET_DIR}/mlst_schema -o ${TARGET_DIR}/marker_seeds/mlst_seeds.tsv

# ====== Known polymorphisms
perform_primer_search "cpsA" "GTAGAAGAAGCAAGCCAGTACGCC" "CCTCTGCAGCAATCTGTTTCATGG" 478
perform_primer_search "cpsB" "GTGTCATCACAGCTATCGTCGC" "CCGGCATTGGATAAGGAAATAGCC" 603
perform_primer_search "cpsC" "CCTGAATATCAATGTATTTGGGCAGTC" "CCAACGCTTTGCTTCTTGAATGAC" 300
perform_primer_search "cpsD" "GGATTCTCTTGTTCAACAAACCATTGG" "CGCATGGCTTCATAAAAGAACAGC" 522
perform_primer_search "cpsE" "GAGGTTGAGCGAGATATATTATGGC" "CACTTCATAAACCGACTCATCACG" 450
perform_primer_search "cpsF" "GCATTACAAGGTTATACAGTTGATGG" "GACTGTTCCATCTTATCTTTTATTCGG" 580
perform_primer_search "cpsG" "GGCTCTGATCAAATGTGGAATCCC" "GGTGTATCTTCAGAAACATATTCTACTG" 503
perform_primer_search "cpsH" "GTGTCTTTAGCAATTGGTATCGGTTG" "CACTAGAGTAGCTAATACTTTTTTTTCCC" 366
perform_primer_search "cpsI" "GCTTGTGAAGCAGCTAAACGAGG" "CTCTGATAAGTAAGTTTCTTTCTCTGCC" 630
perform_primer_search "cpsJ" "CCTCGACGTATATTCTGGAGAAAC" "GCTTAGTTTCACCAAATGCACGTAG" 553
perform_primer_search "cpsK" "GCGTTGCACAACGAATTGCTAAATAC" "CGCTACAATATAGTAAGGTAGCTGAATC" 422


# ======= suspected virulence determinants
perform_primer_search "cylA" "GGTTATGCATCAGATCTCTCAA" "TCTTCAGGTTTAAAATCTGG" 223
perform_primer_search "cylB" "GGAGAATTAGTGTTTAGAGCG" "GCTTCATAACCATTGTTACTATAGAAAC" 522
perform_primer_search "cylM" "AAGATTGTCTGTGCCATGGA" "TACTCACTTCCGGCAACCTT" 159
perform_primer_search "cbh" "CTCATAGGATCCATCACCAACATCAC" "TGGCTGGAATTCACTTTTCAGGCTAT" 580
perform_primer_search "gelE" "TTGTTGGAAGTTCATGTCTA" "TTCATTGACCAGAACAGATT" 1484
perform_primer_search "fsrB" "GCATTGTTATCTATGTCGCCATACC" "GGCTTAGTTCCCACACCATC" 396

# ======= pathogenicity island
perform_primer_search "gls24_like" "GCATTAGATGAGATTGATGGTC" "GCGAGGTTCAGTTTCTTC" 446
perform_primer_search "esp" "CGATAAAGAAGAGAGCGGAG" "GCAAACTCTACATCCACGTC" 539
perform_primer_search "psaA" "CTATTTTGCAGCAAGTGATG" "CGCATAGTAACTATCACCATCTTG" 540

cat ${TARGET_DIR}/marker_seeds/*.tsv > ${MARKER_SEED_INDEX}

chronostrain -c chronostrain.ini \
  make-db \
  -m $MARKER_SEED_INDEX \
  -r $REFSEQ_INDEX \
  -b $BLAST_DB_NAME -bd $BLAST_DB_DIR \
  --min-pct-idty $MIN_PCT_IDTY \
  --ident-threshold 0.998 \
  --threads $NUM_CORES \
  -o $CHRONOSTRAIN_TARGET_JSON
