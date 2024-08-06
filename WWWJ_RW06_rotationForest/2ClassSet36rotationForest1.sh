#!/bin/bash
#SBATCH --job-name=2ClassSet36rotationForest1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=skylake
python 2ClassSet36rotationForest1.py
~