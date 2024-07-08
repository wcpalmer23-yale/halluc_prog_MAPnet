#!/bin/bash

#SBATCH --job-name=check
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00

module load miniconda CUDA/12.1.1 cuDNN/8.9.2.26-CUDA-12.1.1 

conda activate generative
./parallel_imgs_7.sh
