#!/bin/bash
#SBATCH --job-name=100_3_1g_7c
#SBATCH --output=c100_3_1g_7cccurricRun_%j_out.txt
#SBATCH --time=47:59:00
#SBATCH --partition=cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 5 -v python3 -m scripts.trainCurriculum --procs 32 --numCurric 7 --stepsPerCurric 3 --nGen 1 --iterPerEnv 100000 --model NSGA_100k_3step_1gen_7curric --seed 9152

echo "---------- Cluster Job End ---------------------"
