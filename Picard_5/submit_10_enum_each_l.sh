#!/bin/bash
#
#SBATCH --job-name=Pic5_10
#SBATCH --output=logs/res_Pic5_10_%a.out
#SBATCH --error=logs/res_Pic5_10_%a.err
#
#SBATCH --array=1-46
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=2-00:00
#SBATCH --mem=100G

mkdir -p logs results

srun julia --threads 64 enumerate_pseudomanifolds_10_each_l.jl $SLURM_ARRAY_TASK_ID