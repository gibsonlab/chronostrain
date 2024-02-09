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

# 0.00000001 -> clusters = 3622, phylo A clusters = 469
# 0.0013 -> phylo A clusters = 244
# 0.0011 -> clusters = 1845, phylo A clusters = 263
# 0.0012 -> clusters = 1803, phylo A clusters = 255
# 0.0009 -> clusters = 1951, phylo A clusters = 279
# 0.0007 -> clusters = 2070, phylo A clusters = 299
# 0.0005 -> clusters = 2216, phylo A clusters = 321
# 0.0004 -> clusters = 2329, phylo A clusters = 333
# 0.0003 -> clusters = 2468, phylo A clusters = 342
# 0.0001 -> clusters = 2915, phylo A clusters = 387
thresh=0.0003  # empirically tested on ecoli for phylogroup A.
echo "[*] Using database thresholding (Threshhold = ${thresh})."
poppunk --fit-model threshold --ref-db database --threshold ${thresh} --output threshold --threads ${N_CORES}

echo "[*] Done."


# ====================== old code
#echo "[*] Running poppunk model fit (--fit-model) with DBSCAN"
#echo "[**] NOTE: if a Segfault occurs, try running poppunk with 1 thread, which might output a more helpful error message."
#poppunk --fit-model dbscan --ref-db database --output dbscan --threads ${N_CORES}
#
#echo "[*] Running poppunk model fit (--fit-model) refinement"
#echo "[**] NOTE: if a Segfault occurs, try running poppunk with 1 thread, which might output a more helpful error message."
#poppunk --fit-model refine --ref-db database --model-dir dbscan --output refine --threads ${N_CORES}
