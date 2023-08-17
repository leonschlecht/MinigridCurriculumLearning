#!/bin/bash
#SBATCH --job-name=CrMu8080
#SBATCH --output=GA_75_333CrMu8080_%j_out.txt
#SBATCH --time=71:59:00
#SBATCH --partition=cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 trainCurriculum.py --procs 24 --numCurric 3 --stepsPerCurric 3 --nGen 3 --iterPerEnv 75000 --model GA_75k_3step_3gen_3curric_Cr80_Mu80 --mutationProb 0.8 --crossoverProb 0.8 --seed 8515
echo "---------- Cluster Job End ---------------------"
