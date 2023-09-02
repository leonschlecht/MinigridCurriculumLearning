#!/bin/bash
#SBATCH --job-name=dk
#SBATCH --output=GA_dk_%j_out.txt
#SBATCH --time=71:59:00
#SBATCH --partition=cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 trainCurriculum.py --procs 24 --stepsPerCurric 3 --nGen 4 --numCurric 2 --iterPerEnv 100000 --model 6_150k_3step_4gen_2curric --noRewardShaping --seed 9152
echo "---------- Cluster Job End ---------------------"


