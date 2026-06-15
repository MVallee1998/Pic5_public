#!/bin/bash
#
#SBATCH --job-name=Pic5_10_2
#SBATCH --output=res_Pic_5_7-10_2.out
#SBATCH --error=res_Pic_5_7-10_2.err
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=2-00:00
#SBATCH --mem=100G

srun julia --threads 64 enumerate_pseudomanifolds_10_2.jl