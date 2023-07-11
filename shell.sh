#!/usr/bin/env bash

#SBATCH --time=2-05:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --partition=regular

module purge
module load OpenBLAS/0.3.18-GCC-11.2.0

source /home4/s3537307/venv/bin/activate
pypy /home4/s3537307/Scriptie2/main.py