#!/bin/bash
#SBATCH --job-name=constmaxppo8
#SBATCH --output=constmaxppo8_%j_out.txt
#SBATCH --time=23:59:00
#SBATCH --partition=cpu_normal_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 trainCurriculum.py  --trainAllParalell  --model PPO_8x8b --iterPerEnv 50000 --procs 16 --ppoEnv 1 --constMaxsteps --seed 1
srun -c 2 -v python3 trainCurriculum.py  --trainAllParalell  --model PPO_8x8b --iterPerEnv 50000 --procs 16 --ppoEnv 1 --constMaxsteps --seed 1214
srun -c 2 -v python3 trainCurriculum.py  --trainAllParalell  --model PPO_8x8b --iterPerEnv 50000 --procs 16 --ppoEnv 1 --constMaxsteps --seed 2330
srun -c 2 -v python3 trainCurriculum.py  --trainAllParalell  --model PPO_8x8b --iterPerEnv 50000 --procs 16 --ppoEnv 1 --constMaxsteps --seed 9152
echo "---------- Cluster Job End ---------------------"


