#!/bin/bash
#SBATCH --job-name=2ClassSet32xgBoost_3Layer1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=skylake
python 2ClassSet32xgBoost_3Layer1.py
~