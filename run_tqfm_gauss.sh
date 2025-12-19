#!/bin/bash
#SBATCH --job-name=tqfm_moon
#SBATCH --output=/home/%u/logs/tqfm_moon_%j.out
#SBATCH --error=/home/%u/logs/tqfm_moon_%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=8192
#SBATCH --array=1-20

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate svqsvm


python3.13 main_three_losses_parallel_gauss.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 1 --ansatz RealAmplitudes --maxiter 5000


# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 1 --ansatz TwoLocal --optimizer COBYLA --maxiter 50000
# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 2 --ansatz TwoLocal --optimizer COBYLA --maxiter 50000
# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 3 --ansatz TwoLocal --optimizer COBYLA --maxiter 50000

# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 1 --ansatz RealAmplitudes --optimizer COBYLA --maxiter 50000
# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 2 --ansatz RealAmplitudes --optimizer COBYLA --maxiter 50000
# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 3 --ansatz RealAmplitudes --optimizer COBYLA --maxiter 50000

# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 1 --ansatz EfficientSU2 --optimizer COBYLA --maxiter 50000
# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 2 --ansatz EfficientSU2 --optimizer COBYLA --maxiter 50000
# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 3 --ansatz EfficientSU2 --optimizer COBYLA --maxiter 50000

