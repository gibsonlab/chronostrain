#!/bin/bash
set -e
source settings.sh
source msweep/settings.sh

require_program poppunk


poppunk_outdir=${POPPUNK_REFSEQ_DIR}
mkdir -p $poppunk_outdir
cd $poppunk_outdir

echo "[*] Creating target dir ${poppunk_outdir}"
> input.tsv

echo "[*] Adding RefSeq entries to poppunk input."
while IFS=$'\t' read -r genus species strain accession assembly seqpath chromosomelen gffpath
do
  if [ "${accession}" == "Accession" ]; then continue; fi
  if [ "${accession}" == "NZ_CP046226.1" ]; then echo "Skipping NZ_CP046226.1"; continue; fi  # Probably a low-quality assembly. Makes poppunk crash.
  echo "${accession}	${seqpath}" >> input.tsv
done < ${REFSEQ_INDEX}


echo "[*] Running poppunk sketching (--create-db)"
echo "[**] NOTE: if a Segfault occurs, try running poppunk with 1 thread, which might output a more helpful error message."
poppunk --create-db --output database --r-files input.tsv --threads ${N_CORES}

echo "[*] Using database thresholding."
# 0.00000001 -> clusters = 3622, phylo A clusters = 469
thresh=0.00000001  # empirically tested on ecoli for phylogroup A.  0.0013 gives 244 clusters
poppunk --fit-model threshold --ref-db database --threshold ${thresh} --output threshold --threads ${N_CORES}

#echo "[*] Running poppunk model fit (--fit-model) with DBSCAN"
#echo "[**] NOTE: if a Segfault occurs, try running poppunk with 1 thread, which might output a more helpful error message."
#poppunk --fit-model dbscan --ref-db database --output dbscan --threads ${N_CORES}
#
#echo "[*] Running poppunk model fit (--fit-model) refinement"
#echo "[**] NOTE: if a Segfault occurs, try running poppunk with 1 thread, which might output a more helpful error message."
#poppunk --fit-model refine --ref-db database --model-dir dbscan --output refine --threads ${N_CORES}

echo "[*] Done."
