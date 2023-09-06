#!/bin/bash
#SBATCH --job-name=spcl
#SBATCH --output=spcl_%j_out.txt
#SBATCH --time=23:59:00
#SBATCH --partition=cpu_normal_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 trainCurriculum.py  --trainAllParalell  --model SPCL_DynObs2 --iterPerEnv 50000 --procs 16 --seed 1 --allSimultaneous
srun -c 2 -v python3 trainCurriculum.py  --trainAllParalell  --model SPCL_DynObs2 --iterPerEnv 50000 --procs 16 --seed 1214 --allSimultaneous
srun -c 2 -v python3 trainCurriculum.py  --trainAllParalell  --model SPCL_DynObs2 --iterPerEnv 50000 --procs 16 --seed 2330 --allSimultaneous
srun -c 2 -v python3 trainCurriculum.py  --trainAllParalell  --model SPCL_DynObs2 --iterPerEnv 50000 --procs 16 --seed 9152 --allSimultaneous
echo "---------- Cluster Job End ---------------------"


