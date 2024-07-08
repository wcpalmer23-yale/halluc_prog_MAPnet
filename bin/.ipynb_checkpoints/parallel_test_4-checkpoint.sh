#!/bin/bash
export HOME=/home/wcp27/${SLURM_JOB_ID}
for j in $(seq 7567 8286); do
    python utils/gen_images.py --model test/noisy_0.25 --row ${j} \
        --test_type noisy --dval 0.25 --spp 512 # training images clean
done
export HOME=/home/wcp27