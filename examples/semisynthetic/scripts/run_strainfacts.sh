#!/bin/bash
set -e
source settings.sh

n_reads=$1
trial=$2

if [ -z "$n_reads" ]
then
	echo "var \"n_reads\" is empty"
	exit 1
fi

if [ -z "$trial" ]
then
	echo "var \"trial\" is empty"
	exit 1
fi

trial_dir=$(get_trial_dir $n_reads $trial)
read_dir=${trial_dir}/reads
output_dir=${trial_dir}/output/strainfacts

echo "[*] Running StrainFacts inference for n_reads: ${n_reads}, trial: ${trial}"

mkdir -p ${output_dir}
cd ${output_dir}


mg_prefix="mg_all"
metagenotype_all="${mg_prefix}.tsv"

sfacts load --gtpro-metagenotype ${metagenotype_all} ${mg_prefix}.mgen.nc
sfacts fit \
-m ssdd3_with_error  \
--verbose \
--device cuda \
--precision 32 \
--random-seed 0 \
--num-strains 4 \
--nmf-init \
--num-positions 5000 \
--hyperparameters gamma_hyper=1e-10 \
--hyperparameters pi_hyper=0.3 \
--hyperparameters rho_hyper=0.5 \
--hyperparameters alpha_hyper_mean=10.0 \
--hyperparameters alpha_hyper_scale=1e-06 \
--anneal-hyperparameters gamma_hyper=1.0 \
--anneal-hyperparameters rho_hyper=5.0 \
--anneal-hyperparameters pi_hyper=1.0 \
--anneal-steps 10000 --anneal-wait 2000 \
--optimizer-learning-rate 0.05 \
--min-optimizer-learning-rate 1e-06 \
--max-iter 1_000_000 --lag1 50 --lag2 100 \
${mg_prefix}.mgen.nc ${mg_prefix}.world.nc

sfacts dump ${mg_prefix}.world.nc --genotype result_genotypes.tsv --community result_community.tsv
