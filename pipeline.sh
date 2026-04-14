#!/bin/bash
set -ueo pipefail

./scripts/00_download_data.sh

./scripts/01_train_cnn.slurm


