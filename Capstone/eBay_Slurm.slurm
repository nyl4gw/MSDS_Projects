#!/bin/bash
#SBATCH --begin=00:05
#SBATCH --output=result.out
#SBATCH -p standard 
#SBATCH -A nmicp
#SBATCH -t 03:00:00    
#SBATCH --mail-type=fail
#SBATCH --mail-user=fg5t@virginia.edu
module load anaconda 
python eBay_datacollection.py
sbatch eBay_Slurm.slurm 