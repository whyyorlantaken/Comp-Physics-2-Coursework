#!/bin/bash
#SBATCH --job-name=metals
#SBATCH --output=dot.out
#SBATCH --error=dot.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --mem=8G

# Activate the environment
source ~/.bashrc
conda activate cptu

# ------------------------ RUNS ------------------------
python metalconduction.py -p -l -dt 0.2 -t_max 700 -n 8
python metalconduction.py -p -l -dt 0.2 -t_max 700 -n 4
python metalconduction.py -p -l -dt 0.2 -t_max 700 -n 2
python metalconduction.py -p -l -dt 0.2 -t_max 700 -n 1
# ------------------------------------------------------

# Deactivate
conda deactivate
