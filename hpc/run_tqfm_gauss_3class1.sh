#!/bin/bash
#SBATCH --job-name=tqfm_1gauss
#SBATCH --output=/data/%u/logs/tqfm_gauss_%j.out
#SBATCH --error=/data/%u/logs/tqfm_gauss_%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=4096
#SBATCH --array=1-20

# Activate conda environment

export JOB_SCARCH_PATH="/scratch/$SLURM_JOB_ID"
export TMPDIR="$JOB_SCARCH_PATH"


python3.13 parallel_losses_gauss_3class1.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 1 --ansatz EfficientSU2 --maxiter 2000


# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 1 --ansatz TwoLocal --optimizer COBYLA --maxiter 50000
# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 2 --ansatz TwoLocal --optimizer COBYLA --maxiter 50000
# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 3 --ansatz TwoLocal --optimizer COBYLA --maxiter 50000

# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 1 --ansatz RealAmplitudes --optimizer COBYLA --maxiter 50000
# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 2 --ansatz RealAmplitudes --optimizer COBYLA --maxiter 50000
# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 3 --ansatz RealAmplitudes --optimizer COBYLA --maxiter 50000

# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 1 --ansatz EfficientSU2 --optimizer COBYLA --maxiter 50000
# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 2 --ansatz EfficientSU2 --optimizer COBYLA --maxiter 50000
# python3.13 main_tqfm_moon.py --run_id ${SLURM_ARRAY_TASK_ID} --depth 3 --ansatz EfficientSU2 --optimizer COBYLA --maxiter 50000

