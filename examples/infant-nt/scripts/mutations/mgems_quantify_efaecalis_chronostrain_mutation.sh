#!/bin/bash
set -e
source settings.sh
source mgems/settings.sh
shopt -s nullglob

# demix_check should point to demix_check.py (https://github.com/harry-thorpe/demix_check).
# To pass this first check, create a bash executable called "demix_check" that invokes `python demix_check.py` and add to PATH environment var.
require_program demix_check
require_program themisto
require_program mSWEEP
require_program mGEMS
require_program gzip

# ============ Requires arguments:
participant=$1
sample_id=$2
mutation_rate=$3
require_variable 'participant' $participant
require_variable 'sample_id' $sample_id
require_variable 'mutation_rate' $mutation_rate

workdir=$(pwd)
participant_dir=${DATA_DIR}/${participant}
output_dir=${participant_dir}/mgems/${sample_id}
species_breadcrumb=${output_dir}/mgems.species.DONE
breadcrumb=${output_dir}/mgems.efaecalis_mirror.mutation_${mutation_rate}.DONE
if ! [ -f ${species_breadcrumb} ]; then
  echo "[*] mGEMS species-binning for ${participant} [Sample ${sample_id}] not yet done."
  exit 1
fi
if [ -f ${breadcrumb} ]; then
  echo "[*] mGEMS efaecalis quantification (chronostrain mirror, mutated genome 0.${mutation_rate}) for ${participant} [Sample ${sample_id}] already done."
  exit 0
fi


EFAECALIS_CHRONO_MIRROR_REF_DIR=${DATA_DIR}/database/mutated_dbs/${mutation_rate}/mgems
EFAECALIS_CHRONO_MIRROR_REF_INDEX=ref_idx/ref_idx
EFAECALIS_CHRONO_MIRROR_CLUSTER=ref_clu.txt
EFAECALIS_CHRONO_N_COLORS=2375


# ====================================================== script begins here
echo "[*] Running mGEMS efaecalis quantification (chronostrain mirror, mutated genome 0.${mutation_rate}) for ${participant}, sample ${sample_id}"
species_outdir=${output_dir}/species
strain_fq_1=${species_outdir}/binned_reads/Enterococcus_faecalis_1.fastq.gz
strain_fq_2=${species_outdir}/binned_reads/Enterococcus_faecalis_2.fastq.gz

strain_outdir=${output_dir}/Efaecalis_chrono_mutation_${mutation_rate}
strain_aln_1=${strain_outdir}/ali_1.aln
strain_aln_2=${strain_outdir}/ali_2.aln
mkdir -p ${strain_outdir}

cd ${EFAECALIS_CHRONO_MIRROR_REF_DIR}
echo "[**] Aligning fwd+rev reads"
aln_and_compress ${strain_fq_1} ${strain_fq_2} ${strain_aln_1} ${strain_aln_2} ${EFAECALIS_CHRONO_MIRROR_REF_INDEX} ${EFAECALIS_CHRONO_N_COLORS} ${strain_outdir}/tmp

echo "[**] Cleaning up alignment tmpdir."
rm -rf ${strain_outdir}/tmp

min_abun=0.00001
echo "[**] Running mSWEEP abundance estimation (min abundance = ${min_abun})."
mSWEEP \
  -t ${N_CORES} \
  --themisto-1 ${strain_aln_1}  \
  --themisto-2 ${strain_aln_2}  \
  -o ${strain_outdir}/msweep \
  -i ${EFAECALIS_CHRONO_MIRROR_CLUSTER} \
  --bin-reads \
  --min-abundance ${min_abun} \
  --verbose


echo "[**] Extracting reads (for demix_check)."
rm -rf ${strain_outdir}/binned_reads # clear contents, possibly from incomplete previous run
mkdir -p ${strain_outdir}/binned_reads
for bin_file in ${strain_outdir}/*.bin; do
  mv ${bin_file} ${strain_outdir}/binned_reads
done

for bin_file in ${strain_outdir}/binned_reads/*.bin; do
  mGEMS extract --bins ${bin_file} -r ${strain_fq_1},${strain_fq_2} -o ${strain_outdir}/binned_reads
done

echo "[**] Compressing extracted reads."
for f in ${strain_outdir}/binned_reads/*.fastq; do gzip "$f"; done

echo "[**] Running demix_check."
demix_check --mode_check \
  --binned_reads_dir ${strain_outdir}/binned_reads \
  --msweep_abun ${strain_outdir}/msweep_abundances.txt \
  --out_dir ${strain_outdir}/demix_check \
  --ref . \
  --min_abun ${min_abun}

cd ${workdir}
touch "${breadcrumb}"
