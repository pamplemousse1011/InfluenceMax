#!/bin/bash
#SBATCH --job-name=infmax
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00
#SBATCH --mem=16GB
#SBATCH --output=./output/res_%A_%a.out
#SBATCH --error=./output/res_%A_%a.err


python -u ./main.py \
--acquisition_method "infmax" \
--dataset "mnist" \
--available_sample_k 1 \
--max_epochs 20 \
--target_num_acquired_samples 3000 \
--target_accuracy 0.95 \
--batch_size 64 \
--epoch_samples 5056 \
--scoring_batch_size 4096 \
--test_batch_size 16384 \
--validation_set_size 1024 \
--leave_one_out \
--num_models 5 \
--J 1000 \
--seed $SLURM_ARRAY_TASK_ID \

