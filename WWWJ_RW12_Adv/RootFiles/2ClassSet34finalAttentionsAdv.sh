#!/bin/bash
#SBATCH --job-name=2ClassSet34finalAttentionsAdv
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=5:20:00
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=skylake
python 2ClassSet34finalAttentionsAdv.py
~