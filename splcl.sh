#!/bin/bash
#SBATCH --job-name=SPLCL2
#SBATCH --output=SPLCL3_%j.txt
#SBATCH --time=11:59:00
#SBATCH --partition=cadpool_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --verbose
echo "Start2"
srun -c 1 -v python3 -m scripts.trainCurriculum --procs 32 --trainAllParalell --allSimultaneous --model SPLCL --seed 1214
echo "END run"
