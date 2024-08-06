#!/bin/bash
#SBATCH --job-name=8ClassSet35deepForest1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=60G
#SBATCH --partition=skylake
python 8ClassSet35deepForest1.py
~