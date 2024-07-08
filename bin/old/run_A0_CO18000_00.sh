#!/bin/bash
#SBATCH --job-name=A0_CO18000_00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
module load miniconda Julia/1.9.3-linux-x86_64
conda activate generative
./model.sh B A0_CO18000_00 4 4 "[10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]" color 0
