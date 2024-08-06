#!/bin/bash
#SBATCH --job-name=4ClassSet36xgBoost_4Layer1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --partition=skylake
python 4ClassSet36xgBoost_4Layer1.py
~