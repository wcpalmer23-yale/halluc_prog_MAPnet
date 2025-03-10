#!/bin/bash

#SBATCH --job-name=clean_test
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00

module load miniconda Julia/1.9.3-linux-x86_64
conda activate generative2

ttype=$1
dval=$2
proj_dir="/gpfs/radev/home/wcp27/project/halluc_prog_MAPnet/"

mkdir ${proj_dir}/images/test/${ttype}_${dval}
n_imgs=`ls -1 ${proj_dir}/images/test/${ttype}_${dval} | wc -l`
if [[ $n_imgs -lt 9000 ]]; then
    export HOME=/gpfs/radev/scratch/yildirim/wcp27/${SLURM_JOB_ID}
    for j in $(seq 0 8999); do # $(( $n_imgs ))
        python utils/gen_images.py --model test/${ttype}_${dval} --row ${j} \
            --test_type ${ttype} --dval ${dval} --spp 512
    done
    export HOME=/home/wcp27
else
    echo "All images created."
fi
