#!/bin/bash
#SBATCH --job-name=nsga75_3_3_3
#SBATCH --output=nsgac5_75_3_3_3curricRun_%j_out.txt
#SBATCH --time=23:59:00
#SBATCH --partition=cpu_normal_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 1 -v python3 -m scripts.trainCurriculum --procs 32 --numCurric 3 --stepsPerCurric 3 --nGen 3 --iterPerEnv 75000 --model NSGA_75k_3step_3gen_3curric --useNSGA --seed 1214
srun -c 1 -v python3 -m scripts.trainCurriculum --procs 32 --numCurric 3 --stepsPerCurric 3 --nGen 3 --iterPerEnv 75000 --model NSGA_75k_3step_3gen_3curric --useNSGA --seed 2330
srun -c 1 -v python3 -m scripts.trainCurriculum --procs 32 --numCurric 3 --stepsPerCurric 3 --nGen 3 --iterPerEnv 75000 --model NSGA_75k_3step_3gen_3curric --useNSGA --seed 8515
srun -c 1 -v python3 -m scripts.trainCurriculum --procs 32 --numCurric 3 --stepsPerCurric 3 --nGen 3 --iterPerEnv 75000 --model NSGA_75k_3step_3gen_3curric --useNSGA --seed 9152

echo "---------- Cluster Job End ---------------------"

