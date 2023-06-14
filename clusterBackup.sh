#!/bin/bash
#SBATCH --job-name=testRun
#SBATCH --output=OutputSlurm/Wslurm-%j-out.txt
#SBATCH --time=11:59:00
#SBATCH --nodelist=cp2019-02
#SBATCH --partition=cadpool_stud
#SBATCH --exclude=cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --verbose
echo "Start"
srun -c 2 -v python test2.py
