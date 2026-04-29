#!/bin/bash

##Load WM HPC specific modules for working with conda environments
module load miniforge3
source /sciclone/apps/miniforge3-24.9.2-0/etc/profile.d/conda.sh

##Create Conda environment and install necessary packages
mamba create -n kaggleenv -y -c pytorch -c nvidia -c conda-forge \
    python=3.11 \
    pytorch torchvision torchaudio pytorch-cuda=12.1 \
    numpy pillow kaggle

##Activate the environment
conda activate kaggleenv

##Define necessary paths
SHARED_DIR="/sciclone/scr10/gzdata440"
DATA_ROOT="${SHARED_DIR}/fruitsdata2"

##Remove any leftover fruit data from previous incomplete runs
rm -rf "${DATA_ROOT}"
echo "removed fruitsdata2 folder"

##Recreate the folder structure for storing training data
mkdir -p "${DATA_ROOT}"
echo "created fruitsdata2 folder"
cd "${DATA_ROOT}"

##Download the data from github user 'aelchimminut', unzip it and remove leftover junk files
kaggle datasets download aelchimminut/fruits262

unzip fruits262.zip -d "${DATA_ROOT}"

rm fruits262.zip
