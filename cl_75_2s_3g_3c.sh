#!/bin/bash
#SBATCH --job-name=75_2s33
#SBATCH --output=GA_75_2s33c_%j_out.txt
#SBATCH --time=47:59:00
#SBATCH --partition=cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 trainCurriculum.py --procs 24 --numCurric 3 --stepsPerCurric 2 --nGen 3 --iterPerEnv 75000 --model 75k_2step_3gen_3curric --noRewardShaping --seed 9152
echo "---------- Cluster Job End ---------------------"
