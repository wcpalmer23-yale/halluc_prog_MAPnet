#!/bin/bash

#SBATCH --job-name=baseline
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00

module load miniconda CUDA/12.1.1 cuDNN/8.9.2.26-CUDA-12.1.1 

conda activate generative
python inverse_MAP_network/train_MAPnet.py --dataset=baseline --model=new --n_scenes=9 --n_agents=10 --lr=0.0001 --n_epoch=75
