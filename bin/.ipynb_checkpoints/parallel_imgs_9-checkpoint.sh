#!/bin/bash
export HOME=/home/wcp27/${SLURM_JOB_ID}
for j in $(seq 40000 44999); do
    python utils/gen_images.py --model baseline --row ${j} \
        --test_type clean --dval 0.05 --spp 512 # training images clean
done
export HOME=/home/wcp27