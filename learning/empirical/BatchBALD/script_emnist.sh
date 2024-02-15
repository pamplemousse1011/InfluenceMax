#!/bin/bash
#SBATCH --job-name=infmax
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=8GB
#SBATCH --output=./output/res_%A_%a.out
#SBATCH --error=./output/res_%A_%a.err

python -u ./BatchBALD/run_experiment.py \
--dataset "emnist_letters" \
--acquisition_method "multibald" \
--available_sample_k 1 \
--epochs 5 \
--target_num_acquired_samples 250 \
--target_accuracy 0.75 \
--batch_size 512 \
--scoring_batch_size 16384 \
--test_batch_size 16384 \
--epoch_samples 20224 \
--validation_set_size 20800 \
--num_inference_samples 50 \
--seed $SLURM_ARRAY_TASK_ID \


python -u ./BatchBALD/run_experiment.py \
--dataset "emnist_letters" \
--acquisition_method "random" \
--available_sample_k 1 \
--epochs 5 \
--target_num_acquired_samples 250 \
--target_accuracy 0.75 \
--batch_size 512 \
--scoring_batch_size 16384 \
--test_batch_size 16384 \
--epoch_samples 20224 \
--validation_set_size 20800 \
--num_inference_samples 50 \
--seed $SLURM_ARRAY_TASK_ID \


