#!/bin/bash
#SBATCH --job-name=8ClassSet33xgBoost500Estimators1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --partition=skylake
python 8ClassSet33xgBoost500Estimators1.py
~