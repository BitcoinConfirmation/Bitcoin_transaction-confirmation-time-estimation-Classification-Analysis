#!/bin/bash
#SBATCH --job-name=6ClassSet34finalLSTM1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=5:20:00
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=skylake
python 6ClassSet34finalLSTM1.py
~