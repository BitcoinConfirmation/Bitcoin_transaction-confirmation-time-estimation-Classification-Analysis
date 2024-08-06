#!/bin/bash
#SBATCH --job-name=6ClassSet33deepForest1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=60G
#SBATCH --partition=skylake
python 6ClassSet33deepForest1.py
~