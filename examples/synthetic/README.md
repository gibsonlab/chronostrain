A benchmark using synthetic reads to learn abundances of two high-ID strains (5 SNP difference).

To run this example, first navigate to the subdirectory `scripts/read_depths`.

# 1. Create the in silico variants.

```bash
bash create_variants.sh
```

# 2. Generate the synthetic read datasets.

```bash
bash simulate_reads.sh
```

# 3. Run each respective method.

## 3.1 ChronoStrain

```bash
bash filter_all.sh
bash run_chronostrain_all.sh
```

## 3.2 StrainEst

```bash
bash run_strainest_all.sh
```

## 3.3 StrainGST

```bash
bash run_straingst_all.sh
```

Note that we also created a script to run StrainGR, but this produces pileups 
(and not abundance profiles) if run with 1 reference strain, or fails to produce nonzero
abundances for strains not already detected by StrainGST.

# 4. Evaluate the results.

The accuracy/performance metrics are compiled into a table.
```bash
bash evaluate.sh
```
