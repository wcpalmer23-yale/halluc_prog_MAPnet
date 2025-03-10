#!/bin/bash
# Model
#   - runs with clean or distorted images and changes in expectation
#   - example: ./model.sh full 0 4 "[10000, 10000, 10000, 10000, 10000, 10000]" noisy 1
################################################################################################################
# Set directory
proj_dir=/home/wcp27/project/halluc_prog_MAPnet

# Load inputs
iter=$1
begin=$2
end=$3

# Create directory for mitsuba
mkdir /gpfs/radev/scratch/yildirim/wcp27/${SLURM_JOB_ID}

# Help creating images
export HOME=/gpfs/radev/scratch/yildirim/wcp27/${SLURM_JOB_ID}
for j in $(seq $end -1 $begin); do
    python utils/gen_images.py --model ${iter}/baseline \
        --row ${j} --test_type clean --dval 0.03 --spp 512 # training images clean
done
export HOME=/home/wcp27
