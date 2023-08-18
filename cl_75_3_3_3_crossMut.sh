#!/bin/bash
#SBATCH --job-name=sobol
#SBATCH --output=sobol_%j_out.txt
#SBATCH --time=71:59:00
#SBATCH --partition=cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 trainCurriculum.py --procs 24 --numCurric 3 --stepsPerCurric 3 --nGen 3 --iterPerEnv 75000 --model 75k_3step_3gen_3curric_cr9_mu9 --noRewardShaping --crossoverProb .9 --mutationProb .9 --seed 9153
echo "---------- Cluster Job End ---------------------"
