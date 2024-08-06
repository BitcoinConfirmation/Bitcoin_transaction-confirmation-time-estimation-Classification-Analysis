#!/bin/bash
#SBATCH --job-name=8ClassSet32xgBoostMemPos1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=skylake
python 8ClassSet32xgBoostMemPos1.py
~