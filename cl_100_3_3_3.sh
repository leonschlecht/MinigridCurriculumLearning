#!/bin/bash
#SBATCH --job-name=4step
#SBATCH --output=GA_100nRS_%j_out.txt
#SBATCH --time=71:59:00
#SBATCH --partition=cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 trainCurriculum.py --procs 24 --noRewardShaping --numCurric 3 --stepsPerCurric 4 --nGen 3 --iterPerEnv 100000 --model 2_100k_4step_3gen_3curric_gamma70 --crossoverProb .8 --mutationProb .8 --trainingIterations 3500000 --gamma .7 --seed 1214
echo "---------- Cluster Job End ---------------------"


