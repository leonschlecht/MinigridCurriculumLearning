#!/bin/bash
#SBATCH --job-name=c5_150_3_3_3
#SBATCH --output=c5_150_3_3_3_cRun_%j_out.txt
#SBATCH --time=47:59:00
#SBATCH --partition=cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 4 -v python3 -m scripts.trainCurriculum --procs 32 --numCurric 3 --stepsPerCurric 3 --nGen 3 --iterPerEnv 150000 --model NSGA_150k_3step_3gen_3curric --seed 9152
echo "---------- Cluster Job End ---------------------"
