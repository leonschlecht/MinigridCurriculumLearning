#!/bin/bash
#SBATCH --job-name=GA_50_3_3_3
#SBATCH --output=GA_50_3_3_3curricRun_%j_out.txt
#SBATCH --time=47:59:00
#SBATCH --partition=cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=1
#SBATCH --mem=54G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 1 -v python3 -m scripts.trainCurriculum --procs 32 --numCurric 3 --stepsPerCurric 3 --nGen 3 --iterPerEnv 50000 --model 50k_3step_3gen_3curric --seed 1
echo "---------- Cluster Job End ---------------------"

