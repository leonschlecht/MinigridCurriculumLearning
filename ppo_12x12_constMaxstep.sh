#!/bin/bash
#SBATCH --job-name=constmaxppo12x12
#SBATCH --output=constmaxppo12x12_%j_out.txt
#SBATCH --time=23:59:00
#SBATCH --partition=cpu_normal_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 trainCurriculum.py --trainAllParalell --model PPO_12x12b --iterPerEnv 50000 --procs 16 --ppoEnv 3 --constMaxsteps --seed 1
srun -c 2 -v python3 trainCurriculum.py --trainAllParalell --model PPO_12x12b --iterPerEnv 50000 --procs 16 --ppoEnv 3 --constMaxsteps --seed 1214
srun -c 2 -v python3 trainCurriculum.py --trainAllParalell --model PPO_12x12b --iterPerEnv 50000 --procs 16 --ppoEnv 3 --constMaxsteps --seed 2330
srun -c 2 -v python3 trainCurriculum.py --trainAllParalell --model PPO_12x12b --iterPerEnv 50000 --procs 16 --ppoEnv 3 --constMaxsteps --seed 9152
echo "---------- Cluster Job End ---------------------"


