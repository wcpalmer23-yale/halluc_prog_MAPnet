#!/bin/bash

#SBATCH --job-name=Base_A2
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00

module load miniconda Julia/1.9.3-linux-x86_64
conda activate generative
# Creates training images
./baseline_helper.sh A
