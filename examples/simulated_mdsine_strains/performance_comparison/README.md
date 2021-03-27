# Example: Bacterial strains (large test)

Testing chronostrain (with filtering) on sampled reads, using strains from the original MDSINE paper.

## Running
Before running, change variable `PROJECT_DIR` of `run_test.sh` and `plot.sh` to the basepath of this cloned repository 
(located near the top of the file).

### Step 1: `run_test.sh`
`run_test.sh` is run with two command line arguments:
- number of sampled reads
- number of trials

For example:
```
./run_test.sh 1000000 10
```
runs ten independent trials (using different seeds), where each trial samples its own set of one million reads.

### Step 2: `run_test.sh`
To plot the results (after having finished `run_test.sh`), run `plot.sh`, with no arguments.