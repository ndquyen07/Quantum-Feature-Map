#!/bin/bash
#SBATCH --job-name=tqfm_gauss_3class
#SBATCH --output=/home/%u/logs/tqfm_gauss_3class_%j.out
#SBATCH --error=/home/%u/logs/tqfm_gauss_3class_%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4096
#SBATCH --array=1-20

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate svqsvm


python3.13 main_three_losses_parallel_gauss_3class.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 1 --ansatz EfficientSU2 --maxiter 5000


# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 1 --ansatz TwoLocal --optimizer COBYLA --maxiter 50000
# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 2 --ansatz TwoLocal --optimizer COBYLA --maxiter 50000
# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 3 --ansatz TwoLocal --optimizer COBYLA --maxiter 50000

# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 1 --ansatz RealAmplitudes --optimizer COBYLA --maxiter 50000
# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 2 --ansatz RealAmplitudes --optimizer COBYLA --maxiter 50000
# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 3 --ansatz RealAmplitudes --optimizer COBYLA --maxiter 50000

# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 1 --ansatz EfficientSU2 --optimizer COBYLA --maxiter 50000
# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 2 --ansatz EfficientSU2 --optimizer COBYLA --maxiter 50000
# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 3 --ansatz EfficientSU2 --optimizer COBYLA --maxiter 50000

