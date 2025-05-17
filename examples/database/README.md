# What is this directory?

This directory contains jupyter notebooks and helper scripts for the database generation pipeline.

There are multiple ways to arrive at a "valid" database; these scripts were the ones used in our paper.
None of these scripts are "core" to the package; thus we have provided Jupyter notebook recipes as an example (and as an end-to-end pipeline).
In general, the pipeline ought to be modified/customized based on your needs.

We provide 3 recipes:

1) `ecoli_mlst_simple.ipynb` -- a simple, tiny example of an E. coli database with just pubMLST schema-derived markers.
2) `ecoli.ipynb` -- the E. coli pipeline used for our publication. Contains a specialized Serotype PCR primer search.
3) `efaecalis.ipynb` -- a RefSeq analogue of the E. faecalis pipeline used for our publication. Contains a specialized virulance/mutation-island PCR primer search.
