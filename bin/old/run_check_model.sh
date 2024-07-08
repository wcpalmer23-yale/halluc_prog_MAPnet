#!/bin/bash
# Run Check Model
#   - runs with distortions and changes in expectation
#   - example: ./run_check_model 3
################################################################################################################
# Set directory
proj_dir=/home/wcp27/projects/halluc_prog_MAPnet

# Load inputs
iter=$1
test_type=$2
dval=$3
lr=$4
n_epoch=$5

 Loop through models
for i in $(seq 1 ${iter}); do
    if [[ ${i} == 1 ]]; then
        echo RUNNING FIRST LOOP!
        model=baseline

        # Train new model
        eval "$(conda shell.bash hook)"
        conda activate generative
        python train_MAPnet.py --dataset="check_${i}" --model=${model} --lr=${lr} --n_epoch=${n_epoch}

        # Test new model
        python test_MAPnet.py --model="check_${i}" --test_type=${test_type} --dval=${dval}

        # Evaluate new model
        python eval_MAPnet.py --model="check_${i}" --test_type=${test_type}

        # Set model
        model="check_${i}"

        # Deactivate conda
        conda deactivate
    else
        # Generate new training dataset based on predicitons
        alpha=`cat ${proj_dir}/images/${model}/${model}_alpha.txt`
        count=`cat ${proj_dir}/images/${model}/${model}_${test_type}_pred_nagent.txt`
        julia gen_train.jl --dataset "check_${i}" --alpha "${alpha}" --count "${count}"

        # Split new clean training dataset
        eval "$(conda shell.bash hook)"
        conda activate generative
        python split_data.py --dataset="check_${i}"

        # Train new model
        python train_MAPnet.py --dataset="check_${i}" --model=${model} --lr=${lr} --n_epoch=${n_epoch}

        # Test new model
        python test_MAPnet.py --model="check_${i}" --test_type=${test_type} --dval=${dval}

        # Evaluate new model
        python eval_MAPnet.py --model="check_${i}" --test_type=${test_type}

        # Set model
        model="check_${i}"

        # Deactivate conda
        conda deactivate
    fi

done
# Plot trajectories
conda activate generative
python plot_trajectory.py --model="check" --count=${iter} --test_type=${test_type}
