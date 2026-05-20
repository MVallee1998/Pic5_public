#!/bin/bash
#
#SBATCH --job-name=step3
#SBATCH --output=step3_7-10.out
#SBATCH --error=step3_7-10.err
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=2-00:00
#SBATCH --mem=200G

srun julia --threads 64 step3_isom_seed_PLS_10.jl