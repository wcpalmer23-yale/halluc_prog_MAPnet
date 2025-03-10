#!/bin/bash

#SBATCH --job-name=F3
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00

module load miniconda Julia/1.9.3-linux-x86_64
conda activate generative2
# Creates training images
./baseline_helper.sh F 30000 44999
