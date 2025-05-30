#!/bin/bash

#SBATCH -A avijit.d
#SBATCH -n 37
#SBATCH -c 1
#SBATCH --gres=gpu:4
#SBATCH --mem=110000
#SBATCH --time=8-00:00:00
#SBATCH --output=../slurms-outputs/source_only_%j.out

set -e 
eval "$(conda shell.bash hook)"
conda activate domain-adaptation

SEED=1

source_dataset=$1
target_dataset=$2
modality=$3


batch_size=48

mkdir -p results

CUDA_VISIBLE_DEVICES=2 python3 source_only_train.py \
--source_dataset $source_dataset \
--target_dataset $target_dataset \
--modality $modality \
--num_classes 12 \
--adaptation_mode source_only --data_path /home/miguel.oliveira/Workspace/data \
--num_epochs 40 --lr 0.01 \
--batch_size $batch_size --milestone 10 20 \
--weight_decay 1e-7 \
--pretrained ./models/ \
--num_workers 16 \
--split_path ./splits --save_dir ./results \
--seed $SEED \
--gpus 1

# clean all __pycache__ and .pyc files
find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

