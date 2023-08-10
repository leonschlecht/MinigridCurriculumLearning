#!/bin/bash
#SBATCH --job-name=spclDynObs
#SBATCH --output=spcldynobs%j_out.txt
#SBATCH --time=23:59:00
#SBATCH --partition=cpu_normal_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 -m scripts.trainCurriculum  --trainAllParalell  --model SPCL_DynObs --iterPerEnv 50000 --procs 16 --seed 1 --dynamicObstacle --allSimultaneous
srun -c 2 -v python3 -m scripts.trainCurriculum  --trainAllParalell  --model SPCL_DynObs --iterPerEnv 50000 --procs 16 --seed 1214 --dynamicObstacle --allSimultaneous
srun -c 2 -v python3 -m scripts.trainCurriculum  --trainAllParalell  --model SPCL_DynObs --iterPerEnv 50000 --procs 16 --seed 2330 --dynamicObstacle --allSimultaneous
srun -c 2 -v python3 -m scripts.trainCurriculum  --trainAllParalell  --model SPCL_DynObs --iterPerEnv 50000 --procs 16 --seed 9152 --dynamicObstacle --allSimultaneous
echo "---------- Cluster Job End ---------------------"


