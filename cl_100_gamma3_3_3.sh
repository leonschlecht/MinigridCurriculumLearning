#!/bin/bash
#SBATCH --job-name=c100g_70
#SBATCH --output=GA_100Gamma70nRS_%j_out.txt
#SBATCH --time=47:59:00
#SBATCH --partition=cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 -m scripts.trainCurriculum --procs 24 --noRewardShaping --numCurric 3 --stepsPerCurric 3 --nGen 3 --iterPerEnv 100000 --model 100k_3step_3gen_3curric_nRS_gamma50 --gamma .5 --seed 9152
echo "---------- Cluster Job End ---------------------"


