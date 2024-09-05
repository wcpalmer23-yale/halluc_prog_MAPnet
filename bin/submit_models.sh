#!/bin/bash
#SBATCH --job-name=vh_dcm
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --array=26-40
module load miniconda Julia/1.9.3-linux-x86_64
conda activate generative
./run_models.sh E 0 2
