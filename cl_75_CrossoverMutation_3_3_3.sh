#!/bin/bash
#SBATCH --job-name=CrMu5050
#SBATCH --output=GA_75_333CrMu5050_%j_out.txt
#SBATCH --time=47:59:00
#SBATCH --partition=cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=18G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 -m scripts.trainCurriculum --procs 24 --numCurric 3 --stepsPerCurric 3 --nGen 3 --iterPerEnv 75000 --model GA_75k_3step_3gen_3curric --mutationProb 0.5 --crossoverProb 0.5 --seed 1
echo "---------- Cluster Job End ---------------------"
