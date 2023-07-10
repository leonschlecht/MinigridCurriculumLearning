#!/bin/bash
#SBATCH --job-name=n150_32g4c
#SBATCH --output=NSGA_150_32g4c_%j_out.txt
#SBATCH --time=47:59:00
#SBATCH --partition=cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 -m scripts.trainCurriculum --procs 24 --numCurric 4 --stepsPerCurric 3 --nGen 2 --iterPerEnv 150000 --model 150k_3step_4gen_2curric --seed 2330
echo "---------- Cluster Job End ---------------------"
