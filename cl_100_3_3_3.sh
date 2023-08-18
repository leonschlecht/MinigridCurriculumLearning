#!/bin/bash
#SBATCH --job-name=Dync100nRS
#SBATCH --output=GA_100nRS_%j_out.txt
#SBATCH --time=71:59:00
#SBATCH --partition=cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 trainCurriculum.py --procs 24 --noRewardShaping --numCurric 3 --stepsPerCurric 3 --nGen 3 --iterPerEnv 100000 --model 100k_3step_3gen_3curric --crossoverProb .7 --mutationProb .5 --dynamicObstacle --seed 8515
echo "---------- Cluster Job End ---------------------"


