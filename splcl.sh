#!/bin/bash
#SBATCH --job-name=SPLCL3
#SBATCH --output=SPLCL2_%j_out.txt
#SBATCH --time=23:59:00
#SBATCH --partition=cpu_normal_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 -m scripts.trainCurriculum --procs 16 --trainAllParalell --allSimultaneous --model SPLCL_v3noRS --iterPerEnv 25000 --noRewardShaping --seed 1517
echo "---------- Cluster Job End ---------------------"

