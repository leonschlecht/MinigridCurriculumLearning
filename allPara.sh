#!/bin/bash
#SBATCH --job-name=ppo
#SBATCH --output=ppo_%j_out.txt
#SBATCH --time=01:59:00
#SBATCH --partition=cpu_normal_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 -m scripts.trainCurriculum  --trainAllParalell  --model ppoAttempt8x8 --iterPerEnv 50000 --procs 16 --ppoEnv 1 --seed 1
echo "---------- Cluster Job End ---------------------"


