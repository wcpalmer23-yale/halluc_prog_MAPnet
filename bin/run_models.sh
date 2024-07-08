#!/bin/bash
# Run Model
########################################################################################################################################
# Extract inputs
iter=$1
begin=$2
end=$3

# Extract parameters
model=`head -n ${SLURM_ARRAY_TASK_ID} model_params.csv | tail -n 1 | cut -d , -f 1`
dist=`head -n ${SLURM_ARRAY_TASK_ID} model_params.csv | tail -n 1 | cut -d , -f 2`
exp=`head -n ${SLURM_ARRAY_TASK_ID} model_params.csv | tail -n 1 | cut -d , -f 3`

# Run model
./model.sh ${iter} ${model} ${begin} ${end} "[10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]" ${dist} ${exp}
