#!/bin/bash

wget -O HAM10000_part_1.zip 'https://dataverse.harvard.edu/api/access/datafile/3172585?gbrecs=true'
wget -O HAM10000_part_2.zip 'https://dataverse.harvard.edu/api/access/datafile/3172584?gbrecs=true'
mkdir -p training_set

unzip HAM10000_part_1.zip -d training_set
unzip HAM10000_part_2.zip -d training_set

wget -O HAM10000_test.zip 'https://dataverse.harvard.edu/api/access/datafile/3855824?format=original&gbrecs=true'
mkdir -p testing_set
unzip HAM10000_test.zip -d testing_set
mv testing_set/ISIC2018_Task3_Test_Images/* testing_set/
rm -rf testing_set/ISIC2018_Task3_Test_Images

wget -O HAM10000_metadata.csv 'https://dataverse.harvard.edu/api/access/datafile/4338392?format=original&gbrecs=true'
wget -O HAM10000_test_metadata.csv 'https://dataverse.harvard.edu/api/access/datafile/6924466?format=original&gbrecs=true'

rm -rf HAM10000_part_1.zip
rm -rf HAM10000_part_2.zip
rm -rf HAM10000_test.zip

# Following the example from http://chtc.cs.wisc.edu/conda-installation.shtml
# except here we download the installer instead of transferring it
# Download a specific version of Miniconda instead of latest to improve
# reproducibility
export HOME=$PWD
wget -q https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh -O miniconda.sh
sh miniconda.sh -b -p $HOME/miniconda3
rm miniconda.sh
export PATH=$HOME/miniconda3/bin:$PATH

# Set up conda
source $HOME/miniconda3/etc/profile.d/conda.sh
hash -r
conda config --set always_yes yes --set changeps1 no

# Install packages specified in the environment file
conda env create -f environment_preprocess.yml

# Activate the environment and log all packages that were installed
conda activate pytorch-preprocess

python preprocess.py

rm -rf training_set
rm -rf testing_set
rm HAM10000_metadata.csv
rm HAM10000_test_metadata.csv

