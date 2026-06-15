#!/bin/bash
#
#SBATCH --job-name=Pic5_10
#SBATCH --output=logs/res_Pic5_10_26.out
#SBATCH --error=logs/res_Pic5_10_26.err
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=2-00:00
#SBATCH --mem=32G

mkdir -p logs results

srun julia --threads 64 enumerate_pseudomanifolds_10_each_l.jl 26