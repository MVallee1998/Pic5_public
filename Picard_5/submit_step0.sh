#!/bin/bash
#
#SBATCH --job-name=Pic5_10
#SBATCH --output=logs/step1_7-10.out
#SBATCH --error=logs/step1_7-10.err
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00
#SBATCH --mem=30G

srun julia step0_compile_10.jl