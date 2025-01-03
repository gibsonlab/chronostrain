#!/bin/bash
set -e
source settings.sh
source mgems/settings.sh

## Based on the guide from https://github.com/PROBIC/mSWEEP/blob/master/docs/pipeline.md

## create poppunk input
poppunk_outdir=${ECOLI_REF_DIR}/poppunk
mkdir -p ${poppunk_outdir}
cd "${poppunk_outdir}"

echo "[!] Preparing PopPUNK inputs. (${poppunk_outdir})"
grep "\scoli\s" "${REFSEQ_INDEX}" | cut -f4,6 > poppunk_input.tsv  # no need to use sed 1d here, since the grep removes the header.


## run poppunk
echo "[!] Running PopPUNK database construction."
poppunk --create-db --r-files poppunk_input.tsv --threads "${N_CORES}" --output database


## run clustering
# target = 341 phylo A clusters
# threshold = 0.0004, # phylo A clusters = 333
# threhsold = 0.0003, # phylo A clusters = 342
echo "[!] Running PopPUNK clustering."
threshold_value=0.0003
poppunk --fit-model threshold --ref-db database --threshold "${threshold_value}" --output threshold --threads "${N_CORES}"


cd ..
## Create the ref_info.tsv file, using sed+join.
# the first "sed" skips the first line.
# the second "sed" fixes accessions (poppunk replaces the versioning dot in the bio accession with an underscore, so undo it.)
echo "[!] Preparing mGEMS input index files."
echo -e "id\tcluster\tassembly" > ref_info.tsv
join -1 1 -2 1 <(sed '1d' poppunk/threshold/threshold_clusters.csv | sed -E 's/([0-9])_/\1./' | tr ',' '\t' | sort) <(sort poppunk/poppunk_input.tsv) | tr ' ' '\t' >> ref_info.tsv


## Build themisto index.  (IMPORTANT!!! that we use 'cut' to extract directly from ref_paths.txt, to preserve reference ordering in the database between themisto and mSWEEP)
cut -f3 ref_info.tsv | sed '1d' > ref_paths.txt
mkdir -p __tmp
echo "[!] Running themisto build."
themisto build -k 31 -i ref_paths.txt -o "${ECOLI_REF_INDEX}" --temp-dir __tmp


echo "[!] Cleaning up.."
## mSWEEP aux files.
cut -f2 ref_info.tsv | sed '1d' > "${ECOLI_REF_CLUSTER}"

# Optionally, set up demix_check ref (this takes a long time)
#mkdir demix_ref
#cd demix_ref
#setup_reference.sh --ref_info ../ref_info.tsv --threads 2
#cd ../


## clean up
rm -rf __tmp
