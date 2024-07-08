#!/bin/bash
# Model
#   - runs with clean or distorted images and changes in expectation
#   - example: ./model.sh full 0 4 "[10000, 10000, 10000, 10000, 10000, 10000]" noisy 1
################################################################################################################
# Set directory
proj_dir=/home/wcp27/project/halluc_prog_MAPnet

# Load inputs
iter=$1
alpha=$2
lr=$3
n_epoch=$4
n_train=$5
stage=$6

# Create directory for mitsuba
mkdir /gpfs/radev/scratch/yildirim/wcp27/${SLURM_JOB_ID}

# Set necessary inputs
model=new           # loads AlexNet in train_MAPnet.py
count="[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]"

echo ${stage}

if [ "${stage}" = "forward_graphics_engine" ]; then
    echo "FORWARD GRAPHICS ENGINE"
    ## Create dataset
    julia forward_graphics_engine/gen_train.jl --iter ${iter} --dataset baseline \
        --n_train ${n_train} --alpha "${alpha}" --count "${count}"
        
    ## Create images
    n_imgs=`ls -1 ${proj_dir}/images/${iter}/baseline | wc -l`
    if [ -d ${proj_dir}/imagesi/${iter}/baseline/train ] && [ -d ${proj_dir}/images/${iter}/baseline/valid ]; then
        echo "All images created"
    elif [[ $n_train -gt $(( $n_imgs-2  )) ]]; then
        export HOME=/gpfs/radev/scratch/yildirim/wcp27/${SLURM_JOB_ID}
        for j in $(seq $(( $n_imgs-2  )) $(( $n_train-1  ))); do
            python utils/gen_images.py --model ${iter}/baseline \
                --row ${j} --test_type clean --dval 0.05 --spp 512 # training images clean
        done
        export HOME=/home/wcp27
    else
        echo "All images created."
    fi
elif [ "${stage}" = "inverse_MAP_network" ]; then
    echo "INVERSE MAP NETWORK"
    # Split new clean training dataset
    python inverse_MAP_network/split_data.py --iter=${iter} --dataset=baseline
    
    # Train new model
    python inverse_MAP_network/train_MAPnet.py --iter=${iter} --dataset=baseline --model=${model} \
        --n_scenes=9 --n_agents=10 --lr=${lr} --n_epoch=${n_epoch}
    
    # Clean images
    if [ -f ${proj_dir}/images/${iter}/baseline/baseline_classif.pt ] && [ -f ${proj_dir}/images/${iter}/baseline/baseline_conf.pt ]; then
        rm ${proj_dir}/images/${iter}/baseline/train/*.png
        rm ${proj_dir}/images/${iter}/baseline/valid/*.png
    fi
    
    # Clean jit directory
    #rm -r /gpfs/radev/scratch/yildirim/wcp27/${SLURM_JOB_ID}
else
    echo "Stage argument not found! Please choose either forward_graphics_engine or inverse_MAP_network"
fi
