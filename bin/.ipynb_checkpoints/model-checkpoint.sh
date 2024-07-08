#!/bin/bash
# Model
#   - runs with clean or distorted images and changes in expectation
#   - example: ./model.sh full 0 4 "[10000, 10000, 10000, 10000, 10000, 10000]" noisy 1
################################################################################################################
# Set directory
proj_dir=/home/wcp27/project/halluc_prog_MAPnet

# Load inputs
nmodel=$1
start_iter=$2
end_iter=$3
init_alpha=$4
test_type=$5
expect=$6

# Load files
if [ -f ${proj_dir}/lib/${nmodel}/dval.txt ]; then
    readarray -t dval_vals < ${proj_dir}/lib/${nmodel}/dval.txt
else
    echo "${proj_dir}/lib/${nmodel}/dval.txt does not exist.";  exit 1
fi

if [ -f ${proj_dir}/lib/${nmodel}/lr.txt ]; then
    readarray -t lr_vals < ${proj_dir}/lib/${nmodel}/lr.txt
else
    echo "${proj_dir}/lib/${nmodel}/lr.txt does not exist.";  exit 1
fi

if [ -f ${proj_dir}/lib/${nmodel}/n_epoch.txt ]; then
    readarray -t n_epoch_vals < ${proj_dir}/lib/${nmodel}/n_epoch.txt
else
    echo "${proj_dir}/lib/${nmodel}/n_epoch.txt does not exist.";  exit 1
fi

if [ -f ${proj_dir}/lib/${nmodel}/n_train.txt ]; then
    readarray -t n_train_vals < ${proj_dir}/lib/${nmodel}/n_train.txt
else
    echo "${proj_dir}/lib/${nmodel}/n_train.txt does not exist.";  exit 1
fi

if [ -f ${proj_dir}/lib/${nmodel}/conf.txt ]; then
    readarray -t conf_vals < ${proj_dir}/lib/${nmodel}/conf.txt
else
    echo "${proj_dir}/lib/${nmodel}/conf.txt does not exist.";  exit 1
fi


# Check if there are enough specified values
if [[ `echo ${#dval_vals[@]}` -le ${end_iter} ]]; then
    echo "Too few specified dval values."; exit 1
elif [[ `echo ${#lr_vals[@]}` -le ${end_iter} ]]; then
    echo "Too few specified lr values."; exit 1
elif [[ `echo ${#n_epoch_vals[@]}` -le ${end_iter} ]]; then
    echo "Too few specified n_epoch values."; exit 1
elif [[ `echo ${#n_train_vals[@]}` -le ${end_iter} ]]; then
    echo "Too few specified n_train values."; exit 1
elif [[ `echo ${#conf_vals[@]}` -le ${end_iter} ]]; then
    echo "Too few specified conf values."; exit 1
fi

# Loop through models
for i in $(seq ${start_iter} ${end_iter}); do
    # Create directory for mitsuba
    mkdir /home/wcp27/${SLURM_JOB_ID}
    
    # Extract values for iteration
    dval=`echo ${dval_vals[$i]}`
    lr=`echo ${lr_vals[$i]}`
    n_epoch=`echo ${n_epoch_vals[$i]}`
    n_train=`echo ${n_train_vals[$i]}`
    conf=`echo ${conf_vals[$i]}`

    # Set model if continuing previous model
    if [[ ${start_iter} -gt 0 ]]; then
        echo "CONTINUING MODEL"
        model=${nmodel}_$(( $i-1 ))
        ttype_old=${test_type}
    fi

    # Noisy and clean overlap (i.e., test_type=clean := test_type=noisy + dval=0.05)
    if [[ ${test_type} == "noisy" ]] && [[ ${dval} == 0.05 ]]; then
        ttype=clean
    else
        ttype=${test_type}
    fi

    # Generate training dataset
    ## Create csv
    if [[ ${i} == 0 ]] || [[ ${expect} == 0 ]]; then
        model=new           # loads AlexNet in train_MAPnet.py
        alpha=${init_alpha} # uses specified alpha
        count="[0, 0, 0, 0, 0, 0]"
    else
        alpha=`cat ${proj_dir}/images/${model}/${model}_alpha.txt`
        count=`cat ${proj_dir}/images/${model}/${model}_${ttype_old}_pred_nagent.txt`
    fi

    if [[ ${i} == 0 ]] && [[ -f ${proj_dir}/images/baseline/baseline_classif.pt ]]; then
        echo "Copying baseline model."
        cp -r ${proj_dir}/images/baseline ${proj_dir}/images/${nmodel}_0
    else
        julia forward_graphics_engine/gen_train.jl --dataset ${nmodel}_${i} --n_train ${n_train} \
            --alpha "${alpha}" --count "${count}"
        
        ## Create images
        n_imgs=`ls -1 ${proj_dir}/images/${nmodel}_${i} | wc -l`
        if [ -d ${proj_dir}/images/${nmodel}_${i}/train ] && [ -d ${proj_dir}/images/${nmodel}_${i}/valid ]; then
            echo "All images created"
        elif [[ $n_train -gt $(( $n_imgs-2  )) ]]; then
            export HOME=/home/wcp27/${SLURM_JOB_ID}
            for j in $(seq $(( $n_imgs-2  )) $(( $n_train-1  ))); do
                python utils/gen_images.py --model ${nmodel}_${i} --row ${j} \
                    --test_type clean --dval 0.05 --spp 512 # training images clean
            done
            export HOME=/home/wcp27
        else
            echo "All images created."
        fi

        # Split new clean training dataset
        python inverse_MAP_network/split_data.py --dataset=${nmodel}_${i}
    
        # Train new model
        python inverse_MAP_network/train_MAPnet.py --dataset=${nmodel}_${i} --model=${model} \
            --n_scenes=9 --n_agents=10 --lr=${lr} --n_epoch=${n_epoch}
    
        # Clean images
        if [ -f ${proj_dir}/images/${nmodel}_${i}/${nmodel}_${i}_classif.pt ] && [ -f ${proj_dir}/images/${nmodel}_${i}/${nmodel}_${i}_conf.pt ]; then
            rm ${proj_dir}/images/${nmodel}_${i}/train/*.png
            rm ${proj_dir}/images/${nmodel}_${i}/valid/*.png
        fi
    fi

    # Generate test data if doesn't exist
    ## Create csv
    julia forward_graphics_engine/gen_test.jl
    
    ## Create images
    if [[ ${ttype} == "color" ]] || [[ ${ttype} == "edge" ]] || [[ ${ttype} == "complex" ]] || [[ ${ttype} == "cedge" ]] || [[ ${ttype} = "mixed" ]]; then
        n_imgs=`ls -1 ${proj_dir}/images/test/clean_0.05 | wc -l`
        if [[ $n_imgs -lt 9000 ]]; then
            export HOME=/home/wcp27/${SLURM_JOB_ID}
            for j in $(seq $(( $n_imgs  )) 8999); do
                python utils/gen_images.py --model test/clean_0.05 --row ${j} \
                    --test_type clean --dval 0.05 --spp 512
            done
            export HOME=/home/wcp27
        else
            echo "All images created."
        fi
    else
        n_imgs=`ls -1 ${proj_dir}/images/test/${ttype}_${dval} | wc -l`
        if [[ $n_imgs -lt 9000 ]]; then
            export HOME=/home/wcp27/${SLURM_JOB_ID}
            for j in $(seq $(( $n_imgs  )) 8999); do
                python utils/gen_images.py --model test/${ttype}_${dval} --row ${j} \
                    --test_type ${ttype} --dval ${dval} --spp 512
            done
            export HOME=/home/wcp27
        else
            echo "All images created."
        fi
    fi

    # Clean jit home
    rm -r /home/wcp27/${SLURM_JOB_ID}

    # Test new model
    python inverse_MAP_network/test_MAPnet.py --model=${nmodel}_${i} --test_type=${ttype} \
        --n_scenes=9 --n_agents=10 --dval=${dval} --conf=${conf}
    
    # Evaluate new model
    python inverse_MAP_network/eval_MAPnet.py --model=${nmodel}_${i} --test_type=${ttype}

    # Set model
    model=${nmodel}_${i}
    ttype_old=${ttype}
done

# Plot trajectories
python analysis/plot_trajectory.py --model=${nmodel} --count=${end_iter} --test_type=${test_type}
