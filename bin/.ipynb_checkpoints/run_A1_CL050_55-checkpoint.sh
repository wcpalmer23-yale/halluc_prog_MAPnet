#!/bin/bash
#SBATCH --job-name=A1_CL050_55
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00
module load miniconda CUDA/12.1.1 cuDNN/8.9.2.26-CUDA-12.1.1 Julia/1.9.3-linux-x86_64
conda activate generative
./model.sh 0 4 "[10000, 10000, 10000, 10000, 10000, 10000]" clean 1