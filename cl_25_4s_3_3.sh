#!/bin/bash
#SBATCH --job-name=25_4s_3_3
#SBATCH --output=c25_4s_3_3curricRun_%j_out.txt
#SBATCH --time=23:59:00
#SBATCH --partition=cpu_normal_stud,cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 4 -v python3 -m scripts.trainCurriculum --procs 32 --numCurric 3 --stepsPerCurric 4 --nGen 3 --iterPerEnv 25000 --model NSGA_25k_4step_3gen_3curric --seed 9152
echo "---------- Cluster Job End ---------------------"