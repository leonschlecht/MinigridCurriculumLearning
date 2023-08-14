#!/bin/bash
#SBATCH --job-name=APconst
#SBATCH --output=APconst%j_out.txt
#SBATCH --time=23:59:00
#SBATCH --partition=cpu_normal_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 trainCurriculum.py --trainAllParalell  --model APNc --iterPerEnv 50000 --procs 16 --seed 1 --constMaxsteps
srun -c 2 -v python3 trainCurriculum.py --trainAllParalell  --model APNcs --iterPerEnv 50000 --procs 16 --seed 1214 --constMaxsteps
srun -c 2 -v python3 trainCurriculum.py --trainAllParalell  --model APNc --iterPerEnv 50000 --procs 16 --seed 2330 --constMaxsteps
srun -c 2 -v python3 trainCurriculum.py --trainAllParalell  --model APNc --iterPerEnv 50000 --procs 16 --seed 9152 --constMaxsteps
echo "---------- Cluster Job End ---------------------"


