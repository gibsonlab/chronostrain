#!/bin/bash
set -e

TESTNAME="bbvi_test"

cd "/Users/microbiome/Desktop/chronostrain/"
python scripts/simulate_reads.py --seed 123 --out_dir "./data/simulated_reads/bbvi_test" --abundance_path "examples/bbvi_test/strain_abundances.csv" --num_reads 1 --read_length 150

python scripts/run_inference.py --reads_dir "data/simulated_reads/bbvi_test/" --true_abundance_path "data/simulated_reads/bbvi_test/sim_abundances.csv"  --method "bbvi_reparametrization" --read_length 150 --seed 123 --out_dir "data/output/bbvi_test"

