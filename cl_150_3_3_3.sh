#!/bin/bash
#SBATCH --job-name=GA_c5_150_3_3_3
#SBATCH --output=GA_150_3_3_3_cRun_%j_out.txt
#SBATCH --time=23:59:00
#SBATCH --partition=cpu_long_stud,cpu_normal_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 1 -v python3 -m scripts.trainCurriculum --procs 32 --numCurric 3 --stepsPerCurric 3 --nGen 3 --iterPerEnv 150000 --model 150k_3step_3gen_3curric --seed 8515
echo "---------- Cluster Job End ---------------------"
