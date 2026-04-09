#!/bin/bash
module load miniforge3
source /sciclone/apps/miniforge3-24.9.2-0/etc/profile.d/conda.sh
conda create -n kaggleenv -y
conda activate kaggleenv
mamba install kaggle -y

SHARED_DIR="/sciclone/scr10/gzdata440"

mkdir -p "${SHARED_DIR}/fruitsdata"

cd "${SHARED_DIR}/fruitsdata"
kaggle datasets download aelchimminut/fruits262

unzip fruits262.zip -d "${SHARED_DIR}/fruitsdata"

rm fruits262.zip
