#!/bin/bash
#SBATCH --job-name=rrh250
#SBATCH --output=rrh_250_%j_out.txt
#SBATCH --time=47:59:00
#SBATCH --partition=cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 -m scripts.trainCurriculum --noRewardShaping --procs 24 --iterPerEnv 250000 --model rrh_250k_3step_3curric --stepsPerCurric 3 --numCurric 3 --trainRandomRH --seed 3053
echo "---------- Cluster Job End ---------------------"
