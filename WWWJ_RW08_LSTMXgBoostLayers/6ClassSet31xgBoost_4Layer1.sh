#!/bin/bash
#SBATCH --job-name=6ClassSet31xgBoost_4Layer1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=skylake
python 6ClassSet31xgBoost_4Layer1.py
~