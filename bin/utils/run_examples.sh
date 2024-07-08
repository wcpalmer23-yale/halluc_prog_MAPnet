#!/bin/bash

#SBATCH --job-name=examples
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gpus=1
#SBATCH --time=4-00:00:00
#SBATCH --mail-use=william.palmer@yale.edu
#SBATCH --mail-type=ALL

module load miniconda CUDA/12.1.1 cuDNN/8.9.2.26-CUDA-12.1.1 Julia/1.9.3-linux-x86_64

conda activate generative
./gen_examples.sh
