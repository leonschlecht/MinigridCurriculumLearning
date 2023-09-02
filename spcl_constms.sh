#!/bin/bash
#SBATCH --job-name=spclconstms
#SBATCH --output=spclconstmss%j_out.txt
#SBATCH --time=23:59:00
#SBATCH --partition=cpu_normal_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 trainCurriculum.py  --trainAllParalell  --model SPCL_DynObs2 --iterPerEnv 50000 --procs 16 --seed 1 --dynamicObstacle --allSimultaneous --constMaxsteps
srun -c 2 -v python3 trainCurriculum.py  --trainAllParalell  --model SPCL_DynObs2 --iterPerEnv 50000 --procs 16 --seed 1214 --dynamicObstacle --allSimultaneous --constMaxsteps
srun -c 2 -v python3 trainCurriculum.py  --trainAllParalell  --model SPCL_DynObs2 --iterPerEnv 50000 --procs 16 --seed 2330 --dynamicObstacle --allSimultaneous --constMaxsteps
srun -c 2 -v python3 trainCurriculum.py  --trainAllParalell  --model SPCL_DynObs2 --iterPerEnv 50000 --procs 16 --seed 9152 --dynamicObstacle --allSimultaneous --constMaxsteps
echo "---------- Cluster Job End ---------------------"


