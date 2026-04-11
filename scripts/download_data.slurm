#!/bin/bash
#SBATCH --job-name=fruitdownload
#SBATCH --nodes=1 # how many physical machines in the cluster
#SBATCH --ntasks=1 # how many separate 'tasks' (stick to 1)
#SBATCH --cpus-per-task=20 # how many cores (bora max is 20)
#SBATCH --time=180 # d-hh:mm:ss or just No. of minutes
#SBATCH --mem=120G # how much physical memory (all by default)
#SBATCH --mail-type=FAIL,BEGIN,END # when to email you
#SBATCH --mail-user=mrmellors@wm.edu # who to email
#SBATCH -o JOBNAME_%j.out #STDOUT to file (%j is jobID)
#SBATCH -e JOBNAME_%j.err #STDERR to file (%j is jobID)

cd ~/SUPERCOMPUTING_PROJECT/scripts

./00_download_data
