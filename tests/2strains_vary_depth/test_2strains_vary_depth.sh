#!/bin/bash
set -e

TESTNAME="2strains_vary_depth"

cd ../..

for depth in 10 50 100 200 300 400 500
do
  python simulate_reads.py \
  --seed 123 \
  --out_dir "./data/simulated_reads/$TESTNAME/depth_$depth/" \
  --accession_path "tests/$TESTNAME/ncbi_refs.csv" \
  --abundance_path "tests/$TESTNAME/strain_abundances.csv" \
  --num_reads $depth \
  --read_length 150 \
  -trim 2500

  # Time consistency on
  python run_inference.py \
  --read_files \
  "data/simulated_reads/$TESTNAME/depth_$depth/sim_reads_t1.fastq" \
  "data/simulated_reads/$TESTNAME/depth_$depth/sim_reads_t2.fastq" \
  "data/simulated_reads/$TESTNAME/depth_$depth/sim_reads_t4.fastq" \
  "data/simulated_reads/$TESTNAME/depth_$depth/sim_reads_t6.fastq" \
  "data/simulated_reads/$TESTNAME/depth_$depth/sim_reads_t7.fastq" \
  "data/simulated_reads/$TESTNAME/depth_$depth/sim_reads_t8.fastq" \
  "data/simulated_reads/$TESTNAME/depth_$depth/sim_reads_t10.fastq" \
  "data/simulated_reads/$TESTNAME/depth_$depth/sim_reads_t15.fastq" \
  --true_abundance_path "data/simulated_reads/$TESTNAME/depth_$depth/sim_abundances.csv" \
  --accession_path "tests/$TESTNAME/ncbi_refs.csv" \
  --time_points 1 2 4 6 7 8 10 15 \
  --method "em" \
  --seed 123 \
  --out_dir "data/output/test_$TESTNAME/depth_$depth/" \
  --out_file "EM_result_$TESTNAME.csv" \
  --plots_file "EM_result_${TESTNAME}_plot.png" \
  -trim 2500 \
  --time_consistency "on"

   # Time consistency off
  python run_inference.py \
  --read_files \
  "data/simulated_reads/$TESTNAME/depth_$depth/sim_reads_t1.fastq" \
  "data/simulated_reads/$TESTNAME/depth_$depth/sim_reads_t2.fastq" \
  "data/simulated_reads/$TESTNAME/depth_$depth/sim_reads_t4.fastq" \
  "data/simulated_reads/$TESTNAME/depth_$depth/sim_reads_t6.fastq" \
  "data/simulated_reads/$TESTNAME/depth_$depth/sim_reads_t7.fastq" \
  "data/simulated_reads/$TESTNAME/depth_$depth/sim_reads_t8.fastq" \
  "data/simulated_reads/$TESTNAME/depth_$depth/sim_reads_t10.fastq" \
  "data/simulated_reads/$TESTNAME/depth_$depth/sim_reads_t15.fastq" \
  --true_abundance_path "data/simulated_reads/$TESTNAME/depth_$depth/sim_abundances.csv" \
  --accession_path "tests/$TESTNAME/ncbi_refs.csv" \
  --time_points 1 2 4 6 7 8 10 15 \
  --method "em" \
  --seed 123 \
  --out_dir "data/output/test_${TESTNAME}_time_off/depth_$depth/" \
  --out_file "EM_result_$TESTNAME.csv" \
  --plots_file "EM_result_${TESTNAME}_plot.png" \
  -trim 2500 \
  --time_consistency "off"
done
