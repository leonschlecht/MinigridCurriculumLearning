#!/bin/bash
#SBATCH --job-name=rrhDO
#SBATCH --output=debugrrh_DO_%j_out.txt
#SBATCH --time=00:15:00
#SBATCH --partition=cadpool_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 testScript.py
srun -c 2 -v python3 testScript.py
echo "---------- Cluster Job End ---------------------"
