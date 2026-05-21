#!/bin/bash
#
#SBATCH --job-name=Pic5_10
#SBATCH --output=step2_7-10.out
#SBATCH --error=step2_7-10.err
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=2-00:00
#SBATCH --mem=200G

srun julia --threads 64 step2_autom_reduction_10.jl