#!/bin/bash
#SBATCH --job-name=c_50_3s_5g_3c
#SBATCH --output=c50_3s_5g_3c_%j_out.txt
#SBATCH --time=47:59:00
#SBATCH --partition=cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=3
#SBATCH --mem=15G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 3 -v python3 -m scripts.trainCurriculum --procs 32 --numCurric 3 --stepsPerCurric 3 --nGen 5 --iterPerEnv 50000 --model NSGA_50k_3step_5gen_3curric --seed 8515
echo "---------- Cluster Job End ---------------------"
