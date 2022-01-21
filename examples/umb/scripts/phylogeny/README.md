To run, follow these steps:

1. Run ClermonTyping to generate Escherichia clade labels:
    ```bash
   bash run_clermontyping.sh
    ```
2. Align marker genes and concatenate them (to prepare for fasttree).
   ```bash
   bash concatenate_multi_aligns.sh
   ```
3. Create the tree and associated annotations 
   ```bash
   bash create_tree.sh
   ```
   Note: output files will be located inside the phylogeny/ subdirectory of the database. 
4. (Optional) Upload tree and annotations into 