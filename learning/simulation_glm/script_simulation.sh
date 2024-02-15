#!/bin/bash
#SBATCH --job-name=simulation
#SBATCH --cpus-per-task=4
#SBATCH --time=07:30:00
#SBATCH --mem=4GB
#SBATCH --output=./output/res_%A_%a.out
#SBATCH --error=./output/res_%A_%a.err


python -u ./main_sim.py \
--path_logger './logger/' \
--path_output './data-out/' \
--covtype 'PPCA' \
--n_iter 200 \
--n_train 50000 \
--n_test 5000 \
--n_acquire 2000 \
--imax_bsize 1 \
--input_dim 20 \





