#!/bin/bash
#SBATCH --job-name=infmax
#SBATCH --gres=gpu:1
#SBATCH --time=35:00:00
#SBATCH --mem=16GB
#SBATCH --output=./output/res_%A_%a.out
#SBATCH --error=./output/res_%A_%a.err


python -u ./main.py \
--acquisition_method "infmax" \
--dataset "emnist_letters" \
--available_sample_k 1 \
--max_epochs 5 \
--target_num_acquired_samples 2500 \
--target_accuracy 0.75 \
--batch_size 512 \
--epoch_samples 20224 \
--scoring_batch_size 4096 \
--test_batch_size 16384 \
--validation_set_size 20800 \
--leave_one_out \
--num_models 10 \
--J 10 \
--seed $SLURM_ARRAY_TASK_ID \

