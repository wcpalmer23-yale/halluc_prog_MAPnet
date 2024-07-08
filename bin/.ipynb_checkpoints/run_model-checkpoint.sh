#!/bin/bash

#SBATCH --job-name=check
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --gpus=1
#SBATCH --time=4-00:00:00

module load miniconda CUDA/12.1.1 cuDNN/8.9.2.26-CUDA-12.1.1 Julia/1.9.3-linux-x86_64

conda activate generative
#./model_check.sh
./model.sh check 0 2 "[10000, 10000, 10000, 10000, 10000, 10000]" noisy 1
