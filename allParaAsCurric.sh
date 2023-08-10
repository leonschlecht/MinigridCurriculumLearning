#!/bin/bash
#SBATCH --job-name=APDynobsAsCurr
#SBATCH --output=APDynObsAsCurr%j_out.txt
#SBATCH --time=23:59:00
#SBATCH --partition=cpu_normal_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 -m scripts.trainCurriculum  --trainAllParalell  --model AP_Cur_DynObs --iterPerEnv 50000 --procs 16 --seed 1 --dynamicObstacle --asCurriculum
srun -c 2 -v python3 -m scripts.trainCurriculum  --trainAllParalell  --model AP_Cur_DynObs --iterPerEnv 50000 --procs 16 --seed 1214 --dynamicObstacle --asCurriculum
srun -c 2 -v python3 -m scripts.trainCurriculum  --trainAllParalell  --model AP_Cur_DynObs --iterPerEnv 50000 --procs 16 --seed 2330 --dynamicObstacle --asCurriculum
srun -c 2 -v python3 -m scripts.trainCurriculum  --trainAllParalell  --model AP_Cur_DynObs --iterPerEnv 50000 --procs 16 --seed 9152 --dynamicObstacle --asCurriculum
echo "---------- Cluster Job End ---------------------"


