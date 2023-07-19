#!/bin/bash
#SBATCH --job-name=do_CrMuCL5934
#SBATCH --output=GA_75_333CrMu5934_%j_out.txt
#SBATCH --time=47:59:00
#SBATCH --partition=cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 -m scripts.trainCurriculum --procs 24 --noRewardShaping --numCurric 3 --stepsPerCurric 3 --nGen 3 --iterPerEnv 75000 --model GA_75k_3step_3gen_3curric_c59_m34 --mutationProb 0.59 --crossoverProb 0.34 --seed 9152
echo "---------- Cluster Job End ---------------------"
