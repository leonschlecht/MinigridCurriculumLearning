#!/bin/bash
#SBATCH --job-name=c100_3_3_3
#SBATCH --output=c100_3_3_3curricRun_%j_out.txt
#SBATCH --time=23:59:00
#SBATCH --partition=cpu_normal_stud,cpu_long_stud,cadpool_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=3
#SBATCH --mem=20G
#SBATCH --verbose

echo "------------Cluster Job Start-----------------------"

srun -c 3 -v python3 -m scripts.trainCurriculum --procs 48 --numCurric 3 --stepsPerCurric 3 --nGen 3 --iterPerEnv 100000 --model NSGA_100k_3step_3gen_3curric --seed 8515

echo "---------- Cluster Job End ---------------------"
