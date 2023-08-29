Step 1: run `run_poppunk.sh` to cluster the genomes (this occurs independently of any replicate.)
Step 2: run `prepare_all.sh` to create the dbg/kmer index. (a separate dbg is created for each replicate.)
Step 3: run `analyze_all.sh` (perform themisto pseudoalign + msweep inference for each replicate/trial.)

This completes the analysis pipeline.
