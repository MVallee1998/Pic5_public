#!/bin/bash
#
#SBATCH --job-name=Pic5_10
#SBATCH --output=logs/step2_7-10.out
#SBATCH --error=logs/step2_7-10.err
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=2-00:00
#SBATCH --mem=100G

srun julia --threads 32 step2_autom_reduction_10.jl