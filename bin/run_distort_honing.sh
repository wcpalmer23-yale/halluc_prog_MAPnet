#!/bin/bash

#SBATCH --job-name=hone_dist
#SBATCH --partition=day
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00

module load miniconda Julia/1.9.3-linux-x86_64
conda activate generative2

proj_dir=/home/wcp27/project/halluc_prog_MAPnet

nmodel=$1
ttype=$2
dval=$3

# Tests desired model
for iter in A B C D E F G H I J; do
    echo ${iter}
    # Make model directory
    mkdir ${proj_dir}/images/${iter}/${nmodel}_0/
    
    # Copy relavent files
    cp ${proj_dir}/images/${iter}/baseline/baseline_alpha.txt ${proj_dir}/images/${iter}/${nmodel}_0/${nmodel}_0_alpha.txt
    cp ${proj_dir}/images/${iter}/baseline/baseline_classif.pt ${proj_dir}/images/${iter}/${nmodel}_0/${nmodel}_0_classif.pt
    cp ${proj_dir}/images/${iter}/baseline/baseline_conf.pt ${proj_dir}/images/${iter}/${nmodel}_0/${nmodel}_0_conf.pt
    cp ${proj_dir}/images/${iter}/baseline/baseline_training_loss.csv ${proj_dir}/images/${iter}/${nmodel}_0/${nmodel}_0_training_loss.csv
    cp ${proj_dir}/images/${iter}/baseline/labels.csv ${proj_dir}/images/${iter}/${nmodel}_0/labels.csv
    
    python inverse_MAP_network/test_MAPnet.py --iter=${iter} --model=${nmodel}_0 --test_type=${ttype} --n_scenes=9 --n_agents=10 --dval=${dval} --conf=0
    python inverse_MAP_network/eval_MAPnet.py --iter=${iter} --model=${nmodel}_0 --test_type=${ttype}
done
