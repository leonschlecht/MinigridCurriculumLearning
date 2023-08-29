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
srun -c 2 -v python3 trainCurriculum.py --procs 24 --stepsPerCurric 2 --nGen 4 --numCurric 3 --iterPerEnv 100000 --model 5_100k_2step_4gen_3curric --crossoverProb 0.6 --mutationProb 0.6 --noRewardShaping --seed 8515
echo "---------- Cluster Job End ---------------------"


