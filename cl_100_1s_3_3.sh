#!/bin/bash
#SBATCH --job-name=noRHCL
#SBATCH --output=GAcHIGHERnorhclS_%j_out.txt
#SBATCH --time=71:59:00
#SBATCH --partition=cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 trainCurriculum.py --procs 24 --noRewardShaping --numCurric 2 --stepsPerCurric 2 --nGen 5 --iterPerEnv 100000 --model 100k_2step_5gen_2curric --crossoverProb .8 --mutationProb .8 --seed 1214
echo "---------- Cluster Job End ---------------------"


