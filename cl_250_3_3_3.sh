#!/bin/bash
#SBATCH --job-name=moGA250_333
#SBATCH --output=GA_250_333_%j_out.txt
#SBATCH --time=47:59:00
#SBATCH --partition=cpu_normal_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 -m scripts.trainCurriculum --procs 24 --numCurric 3 --stepsPerCurric 3 --nGen 3 --iterPerEnv 250000 --model 250k_3step_3gen_3curric --seed 2330
echo "---------- Cluster Job End ---------------------"
