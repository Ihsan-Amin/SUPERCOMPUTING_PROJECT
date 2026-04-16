#!/bin/bash
#SBATCH --job-name=fruit_cnn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00          # 12 hours (3 models sequentially)
#SBATCH --mem=64G
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=mrmellors@wm.edu
#SBATCH -o /sciclone/scr10/gzdata440/fruitsdata/logs/fruit_cnn_%j.out
#SBATCH -e /sciclone/scr10/gzdata440/fruitsdata/logs/fruit_cnn_%j.err
#SBATCH --gpus=1

./scripts/00_download_data.sh

./scripts/01_train_cnn.slurm


