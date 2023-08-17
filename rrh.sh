#!/bin/bash
#SBATCH --job-name=rrhDO
#SBATCH --output=rrh_DO_%j_out.txt
#SBATCH --time=47:59:00
#SBATCH --partition=cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 trainCurriculum.py --noRewardShaping --procs 24 --iterPerEnv 100000 --model rrh_100k_3_3 --stepsPerCurric 3 --numCurric 3 --trainRandomRH --seed 9152 --dynamicObstacle
echo "---------- Cluster Job End ---------------------"
