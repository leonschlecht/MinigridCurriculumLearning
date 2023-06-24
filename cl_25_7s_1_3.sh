#!/bin/bash
#SBATCH --job-name=c_25_7s_1_3
#SBATCH --output=c25_7s_1_3_%j_out.txt
#SBATCH --time=47:59:00
#SBATCH --partition=cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 4 -v python3 -m scripts.trainCurriculum --procs 96 --numCurric 3 --stepsPerCurric 7 --nGen 1 --iterPerEnv 25000 --model NSGA_25k_7step_1gen_3curric --seed 2330
echo "---------- Cluster Job End ---------------------"
