#!/bin/bash
#SBATCH --job-name=metals
#SBATCH --output=dot.out
#SBATCH --error=dot.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00
#SBATCH --mem=8G

# Activate the environment
source ~/.bashrc
conda activate cptu

# --------------- PARALLEL RUNS ---------------
python metalconduction.py -p -l -m Copper
python metalconduction.py -p -l -m Copper Iron
python metalconduction.py -p -l -m Copper Iron Aluminum Brass
python metalconduction.py -p -l -m Copper Iron Aluminum Brass Steel Zinc Lead Titanium