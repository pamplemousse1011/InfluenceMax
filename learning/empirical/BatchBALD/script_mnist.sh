#!/bin/bash
#SBATCH --job-name=infmax
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=8GB
#SBATCH --output=./output/res_%A_%a.out
#SBATCH --error=./output/res_%A_%a.err

python -u ./BatchBALD/run_experiment.py \
--dataset "mnist" \
--acquisition_method "multibald" \
--available_sample_k 1 \
--epochs 20 \
--target_num_acquired_samples 5000 \
--target_accuracy 0.95 \
--batch_size 64 \
--scoring_batch_size 16384 \
--test_batch_size 16384 \
--epoch_samples 5056 \
--validation_set_size 1024 \
--num_inference_samples 100 \
--seed $SLURM_ARRAY_TASK_ID \


python -u ./BatchBALD/run_experiment.py \
--dataset "mnist" \
--acquisition_method "random" \
--available_sample_k 1 \
--epochs 20 \
--target_num_acquired_samples 5000 \
--target_accuracy 0.95 \
--batch_size 64 \
--scoring_batch_size 16384 \
--test_batch_size 16384 \
--epoch_samples 5056 \
--validation_set_size 1024 \
--num_inference_samples 100 \
--seed $SLURM_ARRAY_TASK_ID \

