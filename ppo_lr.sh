#!/bin/bash
#SBATCH --job-name=constppoLR
#SBATCH --output=ppoLRconst_%j_out.txt
#SBATCH --time=23:59:00
#SBATCH --partition=cpu_normal_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 trainCurriculum.py --trainAllParalell  --model PPOb_12x12b_lr01 --iterPerEnv 50000 --procs 16 --ppoEnv 3 --seed 1 --lr .01 --constMaxsteps
srun -c 2 -v python3 trainCurriculum.py --trainAllParalell  --model PPOb_12x12b_lr003 --iterPerEnv 50000 --procs 16 --ppoEnv 3 --seed 1214 --lr .003 --constMaxsteps
srun -c 2 -v python3 trainCurriculum.py --trainAllParalell  --model PPOb_12x12b_lr0001 --iterPerEnv 50000 --procs 16 --ppoEnv 3 --seed 2330 --lr .0001 --constMaxsteps
srun -c 2 -v python3 trainCurriculum.py --trainAllParalell  --model PPOb_12x12b_lr00001 --iterPerEnv 50000 --procs 16 --ppoEnv 3 --seed 9152 --lr .00001 --constMaxsteps
echo "---------- Cluster Job End ---------------------"


