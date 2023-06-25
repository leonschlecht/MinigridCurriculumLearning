#!/bin/bash
#SBATCH --job-name=c_25_2_4_4
#SBATCH --output=c25_7s_1_3_%j_out.txt
#SBATCH --time=47:59:00
#SBATCH --partition=cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=3
#SBATCH --mem=10G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 3 -v python3 -m scripts.trainCurriculum --procs 96 --numCurric 4 --stepsPerCurric 2 --nGen 4 --iterPerEnv 25000 --model NSGA_25k_2step_4gen_4curric --seed 8515
echo "---------- Cluster Job End ---------------------"
