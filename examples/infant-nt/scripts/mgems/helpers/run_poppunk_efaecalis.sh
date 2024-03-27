#!/bin/bash
source settings.sh
source mgems/settings.sh
set -e

ref_dir=$1
infant_index=$2
thresh=$3

N_CORES=12
poppunk_outdir=${ref_dir}/poppunk
mkdir -p $poppunk_outdir
cd $poppunk_outdir
> input.tsv

echo "[*] Adding European Isolate Assemblies to poppunk input."
while IFS=$'\t' read -r genus species strain accession assembly seqpath chromosomelen gffpath
do
  if [ "${species}" == "faecalis" ]; then
    echo -e "${accession}\t${seqpath}" >> input.tsv
  fi
done < ${EUROPE_EFAECALIS_INDEX}


echo "[*] Adding Infant isolate assemblies to poppunk input."
while IFS=$'\t' read -r genus species strain infant_id time_point accession assembly seqpath chromosomelen gffpath
do
  if [ "${species}" == "faecalis" ]; then
    echo -e "${accession}\t${seqpath}" >> input.tsv
  fi
done < ${infant_index}


echo "[*] Running poppunk sketching (--create-db)"
poppunk --create-db --output database --r-files input.tsv --threads ${N_CORES}

echo "[*] Using database thresholding (Threshhold = ${thresh})."
poppunk --fit-model threshold --ref-db database --threshold ${thresh} --output threshold --threads ${N_CORES}

echo "[*] Done."
