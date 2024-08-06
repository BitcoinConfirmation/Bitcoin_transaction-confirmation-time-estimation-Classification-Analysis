#!/bin/bash
#SBATCH --job-name=2ClassSet31Transformer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=6:20:00
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=skylake
python 2ClassSet31Transformer.py
~