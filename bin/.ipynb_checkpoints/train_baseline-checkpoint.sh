#!/bin/bash
echo "Training"
python inverse_MAP_network/train_MAPnet.py --dataset=baseline --model=new --n_scenes=9 --n_agents=10 --lr=0.0001 --n_epoch=75