# microbiome_tracking

Required packages:

- PyTorch
- Seaborn
- Joblib
- tqdm
- BioPython


Accession reference CSV file example (test/ncbi_refs.json):

--------------------------------
```
[
    {
        "markers": [
                    {"type":"tag", "id":"BF9343_0009", "name":"glycosyltransferase"}, 
                    {"type":"tag", "id":"BF9343_3272", "name":"hydrolase"}
                ], 
        "name": "bacteroides fargilis", 
        "accession": "CR626927.1"
    },
    {
        "markers": {},
        "name": "bacteriodes thetaiotaomicron",
        "accession": "NZ_CP012937.1"
    },
    {
        "markers": [
                    {"type":"tag", "id":"ECOLIN_01070", "name":"16S"},
                    {"type": "primer", "forward": "GTGCCAGCMGCCGCGGTAA", "reverse": "GGACTACHVGGGTWTCTAAT", "name": "16S_V4"}
                ],
        "name": "escherichia coli nissle 1917",
        "accession": "CP007799.1"
    }
]
```
--------------------------------
