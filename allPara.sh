#!/bin/bash
#SBATCH --job-name=ppo10x10
#SBATCH --output=ppo10x10_%j_out.txt
#SBATCH --time=23:59:00
#SBATCH --partition=cpu_normal_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 -m scripts.trainCurriculum  --trainAllParalell  --model PPO_10x10 --iterPerEnv 50000 --procs 16 --ppoEnv 2 --seed 3053
srun -c 2 -v python3 -m scripts.trainCurriculum  --trainAllParalell  --model PPO_10x10 --iterPerEnv 50000 --procs 16 --ppoEnv 2 --seed 8258
srun -c 2 -v python3 -m scripts.trainCurriculum  --trainAllParalell  --model PPO_10x10 --iterPerEnv 50000 --procs 16 --ppoEnv 2 --seed 9152
srun -c 2 -v python3 -m scripts.trainCurriculum  --trainAllParalell  --model PPO_10x10 --iterPerEnv 50000 --procs 16 --ppoEnv 2 --seed 8515
srun -c 2 -v python3 -m scripts.trainCurriculum  --trainAllParalell  --model PPO_10x10 --iterPerEnv 50000 --procs 16 --ppoEnv 2 --seed 1
echo "---------- Cluster Job End ---------------------"


