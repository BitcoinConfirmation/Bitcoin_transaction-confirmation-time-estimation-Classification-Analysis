#!/bin/bash
#SBATCH --job-name=4ClassSet33xgBoost_0Layer1000Estimators1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --partition=skylake
python 4ClassSet33xgBoost_0Layer1000Estimators1.py
~