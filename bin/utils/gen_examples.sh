#!/bin/bash
# Generates examples of distorted iamges
################################################################################################################
# Set directory
proj_dir=/home/wcp27/project/halluc_prog_MAPnet
mkdir /home/wcp27/${SLURM_JOB_ID}
export HOME=/home/wcp27/${SLURM_JOB_ID}
## Noisy
ttype=noisy
dvals=(0.05 0.15 0.25 0.35 0.45 0.55)
for dval in ${dvals[@]}; do
    for j in $(seq 0 191); do
        python gen_images.py --model examples/${ttype}_${dval} --row ${j} \
            --test_type ${ttype} --dval ${dval} --spp 512
    done
done

## Blurred
ttype=blurred
dvals=(0.0 2.0 4.0 6.0 8.0 10.0)
for dval in ${dvals[@]}; do
    for j in $(seq 0 191); do
        python gen_images.py --model examples/${ttype}_${dval} --row ${j} \
            --test_type ${ttype} --dval ${dval} --spp 512
    done
done

