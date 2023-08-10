#!/bin/bash
#SBATCH --job-name=ppoLR
#SBATCH --output=ppoLR_%j_out.txt
#SBATCH --time=23:59:00
#SBATCH --partition=cpu_normal_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 -m scripts.trainCurriculum  --trainAllParalell  --model PPO_12x12b_lr01 --iterPerEnv 50000 --procs 16 --ppoEnv 3 --seed 1 --lr .01
srun -c 2 -v python3 -m scripts.trainCurriculum  --trainAllParalell  --model PPO_12x12b_lr003 --iterPerEnv 50000 --procs 16 --ppoEnv 3 --seed 1214 --lr .003
srun -c 2 -v python3 -m scripts.trainCurriculum  --trainAllParalell  --model PPO_12x12b_lr0001 --iterPerEnv 50000 --procs 16 --ppoEnv 3 --seed 2330 --lr .0001
srun -c 2 -v python3 -m scripts.trainCurriculum  --trainAllParalell  --model PPO_12x12b_lr00001 --iterPerEnv 50000 --procs 16 --ppoEnv 3 --seed 9152 --lr .00001
echo "---------- Cluster Job End ---------------------"


