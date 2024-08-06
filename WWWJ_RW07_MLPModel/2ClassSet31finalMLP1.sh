#!/bin/bash
#SBATCH --job-name=31finalMLP1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:20:00
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=skylake
python 2ClassSet31finalMLP1.py
~