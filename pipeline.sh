#!/bin/bash
#SBATCH --job-name=fruit_cnn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00          # 48 hours mainly because deleting any reference to previous data takes an obnoxiously long time, training 3 CNNs sequentially also takes a long time
#SBATCH --mem=64G
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=mrmellors@wm.edu
#SBATCH -o ./logs/fruit_cnn_%j.out
#SBATCH -e ./logs/fruit_cnn_%j.err
#SBATCH --gpus=2

##Download the data from kaggle user aelchimminut and create a conda environment with these packages: pytorch torchvision torchaudio numpy pillow kaggle, see .yml file for more specific information
./scripts/00_download_data.sh

##Train 3 separate CNN models on the downloaded fruits data, alexnet, alexnet_bn, resnet50. Also calculates each models accuracy in terms of classifying groups and compares the models with different metrics
##See 01_train_cnn.py and readme for additional explanation
./scripts/01_train_cnn.slurm
