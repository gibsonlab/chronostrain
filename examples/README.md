This directory contains some small demo files.

# `colab_demo`

A Google Colab notebook which installs the software, and runs a small example.

# `database`

A collection of jupyter notebooks and python/shell scripts for setting up the database.

# `example_configs`
A collection of configuration files meant to serve as examples/templates.
1. chronostrain.ini: A basic configuration file for chronostrain.
2. log_config.ini: A basic logging configuration file for chronostrain.
3. entero_ecoli.json: A sample database JSON file. (Note that the JSON file on its own is not enough; you also need the genome FASTA files!)
4. ecoli.clusters.txt: A sample cluster file.

Note that #3 and #4 are NOT items meant to be manually generated. Please refer to the CLI interface `chronostrain make-db` and `chronostrain cluster-db`.
