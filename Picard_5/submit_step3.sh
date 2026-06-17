#!/bin/bash
#
#SBATCH --job-name=step3
#SBATCH --output=logs/step3_7-10.out
#SBATCH --error=logs/step3_7-10.err
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=2-00:00
#SBATCH --mem=100G

srun julia --threads 32 step3_isom_seed_PLS_10.jl