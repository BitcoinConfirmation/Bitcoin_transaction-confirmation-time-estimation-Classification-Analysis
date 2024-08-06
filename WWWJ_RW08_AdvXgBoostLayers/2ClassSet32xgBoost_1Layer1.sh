#!/bin/bash
#SBATCH --job-name=2ClassSet32xgBoost_1Layer1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --partition=skylake
python 2ClassSet32xgBoost_1Layer1.py
~