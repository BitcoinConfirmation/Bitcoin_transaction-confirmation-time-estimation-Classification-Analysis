#!/bin/bash
#SBATCH --job-name=Test_2ClassSet31xgBoost_3Layer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:30:00
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=skylake


python Test_2ClassSet31xgBoost_3Layer.py
~