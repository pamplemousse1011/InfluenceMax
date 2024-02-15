#!/bin/bash
#SBATCH --job-name=infmax
#SBATCH --gres=gpu:1
#SBATCH --time=33:00:00
#SBATCH --mem=16GB
#SBATCH --output=./output/res_%A_%a.out
#SBATCH --error=./output/res_%A_%a.err


python -u ./main.py \
--acquisition_method "infmax" \
--dataset "tweet" \
--available_sample_k 1 \
--target_num_acquired_samples 2000 \
--target_accuracy 0.74 \
--batch_size 512 \
--epoch_samples 101120 \
--scoring_batch_size 4096 \
--test_batch_size 16384 \
--validation_set_size 1024 \
--max_epochs 2 \
--num_models 10 \
--J 50 \
--seed $SLURM_ARRAY_TASK_ID \

