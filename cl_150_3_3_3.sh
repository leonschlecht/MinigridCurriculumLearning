#!/bin/bash
#SBATCH --job-name=150_333cons
#SBATCH --output=150333_constms_%j_out.txt
#SBATCH --time=71:59:00
#SBATCH --partition=cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 trainCurriculum.py --procs 24 --numCurric 3 --stepsPerCurric 3 --nGen 3 --iterPerEnv 150000 --model 150k_3step_3gen_3curric --seed 8515 --constMaxsteps
echo "---------- Cluster Job End ---------------------"
