#!/bin/bash
set -e

TESTNAME="8strains_vary_depth"

cd ../..

for depth in 10 50 100 200 300 400 500
do
  python scripts/simulate_reads.py \
  --seed 123 \
  --out_dir "./data/simulated_reads/$TESTNAME/depth_$depth/" \
  --accession_path "tests/$TESTNAME/ncbi_refs.csv" \
  --abundance_path "tests/$TESTNAME/strain_abundances.csv" \
  --num_reads $depth \
  --read_length 150 \
  -trim 2500

  # Time consistency on
  python scripts/run_inference.py \
  --read_files \
  "data/simulated_reads/$TESTNAME/depth_$depth/sim_reads_t1.fastq" \
  "data/simulated_reads/$TESTNAME/depth_$depth/sim_reads_t2.fastq" \
  "data/simulated_reads/$TESTNAME/depth_$depth/sim_reads_t3.fastq" \
  "data/simulated_reads/$TESTNAME/depth_$depth/sim_reads_t4.fastq" \
  "data/simulated_reads/$TESTNAME/depth_$depth/sim_reads_t5.fastq" \
  "data/simulated_reads/$TESTNAME/depth_$depth/sim_reads_t10.fastq" \
  "data/simulated_reads/$TESTNAME/depth_$depth/sim_reads_t15.fastq" \
  --true_abundance_path "data/simulated_reads/$TESTNAME/depth_$depth/sim_abundances.csv" \
  --accession_path "tests/$TESTNAME/ncbi_refs.csv" \
  --time_points 1 2 3 4 5 10 15 \
  --method "em" \
  --seed 123 \
  --out_path "data/output/$TESTNAME/depth_$depth/EM_result_$TESTNAME.csv" \
  --plots_path "data/output/$TESTNAME/depth_$depth/EM_result_${TESTNAME}_plot.png" \
  -trim 2500

   # Time consistency off
  python scripts/run_inference.py \
  --read_files \
  "data/simulated_reads/$TESTNAME/depth_$depth/sim_reads_t1.fastq" \
  "data/simulated_reads/$TESTNAME/depth_$depth/sim_reads_t2.fastq" \
  "data/simulated_reads/$TESTNAME/depth_$depth/sim_reads_t3.fastq" \
  "data/simulated_reads/$TESTNAME/depth_$depth/sim_reads_t4.fastq" \
  "data/simulated_reads/$TESTNAME/depth_$depth/sim_reads_t5.fastq" \
  "data/simulated_reads/$TESTNAME/depth_$depth/sim_reads_t10.fastq" \
  "data/simulated_reads/$TESTNAME/depth_$depth/sim_reads_t15.fastq" \
  --true_abundance_path "data/simulated_reads/$TESTNAME/depth_$depth/sim_abundances.csv" \
  --accession_path "tests/$TESTNAME/ncbi_refs.csv" \
  --time_points 1 2 3 4 5 10 15 \
  --method "em" \
  --seed 123 \
  --out_path "data/output/test_${TESTNAME}_time_off/depth_$depth/EM_result_$TESTNAME.csv" \
  --plots_path "data/output/test_${TESTNAME}_time_off/depth_$depth/EM_result_${TESTNAME}_plot.png" \
  -trim 2500 \
  --disable_time_consistency
done
