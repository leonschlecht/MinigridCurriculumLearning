#!/bin/bash
#SBATCH --job-name=rrhC6
#SBATCH --output=rrh_100_%j_out.txt
#SBATCH --time=23:59:00
#SBATCH --partition=cpu_normal_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 -m scripts.trainCurriculum --noRewardShaping --procs 24 --iterPerEnv 100000 --model rrh_100k_3step_6curric --stepsPerCurric 3 --numCurric 6 --trainRandomRH --seed 8515

echo "---------- Cluster Job End ---------------------"
