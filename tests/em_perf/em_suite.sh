#!/bin/bash
set -e

TESTNAME="em_perf"
SPARSE_DEPTH="10"

#for depth in 10 100 200 300 400 500 600 700 800 900 1000 1500 2000 3000
for depth in 10 100 200 300 400 500 600 700 800 900 1000
do
  for trial in 1 2 3 4 5 6 7 8 9 10
  do
    python simulate_reads.py \
    --out_dir "./data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/" \
    --accession_path "tests/$TESTNAME/ncbi_refs.csv" \
    --abundance_path "tests/$TESTNAME/true_abundances.csv" \
    --num_reads $depth \
    --read_length 150 \
    -trim 500

#--num_reads $depth $depth $depth $SPARSE_DEPTH $depth $depth $SPARSE_DEPTH $depth \

    # Time consistency on
    python run_inference.py \
    --read_files \
    "data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t1.fastq" \
    "data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t2.fastq" \
    "data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t3.fastq" \
    "data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t4.fastq" \
    "data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t5.fastq" \
    "data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t10.fastq" \
    "data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t11.fastq" \
    "data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_reads_t17.fastq" \
    --true_abundance_path "data/simulated_reads/$TESTNAME/depth_$depth-trial_$trial/sim_abundances.csv" \
    --accession_path "tests/$TESTNAME/ncbi_refs.csv" \
    --time_points 1 2 3 4 5 10 11 17 \
    --method "em" \
    --out_path "data/output/test_$TESTNAME/depth_$depth-trial_$trial/EM_result_$TESTNAME.csv" \
    --plots_path "data/output/test_$TESTNAME/depth_$depth-trial_$trial/EM_result_${TESTNAME}_plot.png" \
    -trim 500 \
    --iters 50000
  done
done