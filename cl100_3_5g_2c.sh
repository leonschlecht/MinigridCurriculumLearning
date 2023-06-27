#!/bin/bash
#SBATCH --job-name=100_3_5g_2c
#SBATCH --output=c100_3_5g_2ccurricRun_%j_out.txt
#SBATCH --time=47:59:00
#SBATCH --partition=cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 4 -v python3 -m scripts.trainCurriculum --procs 32 --numCurric 2 --stepsPerCurric 3 --nGen 5 --iterPerEnv 100000 --model NSGA_100k_3step_5gen_2curric --seed 9152

echo "---------- Cluster Job End ---------------------"
