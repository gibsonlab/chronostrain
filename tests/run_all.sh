#!/bin/bash

cd 2strains
sh test_2strains.sh
cd ../8strains
sh test_8strains.sh
cd ../2strains_vary_depth
sh 2strains_vary_depth/test_2strains_2strains_vary_depth.sh

