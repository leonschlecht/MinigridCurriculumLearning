#!/bin/bash
#SBATCH --job-name=SPLCL
#SBATCH --output=SPLCL2_%j.txt
#SBATCH --time=23:59:00
#SBATCH --partition=cpu_normal_stud,cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --verbose
echo "Start2"
srun -c 1 -v python3 -m scripts.trainCurriculum --procs 32 --trainAllParalell --allSimultaneous --seed 1 --model AllPara
srun -c 1 -v python3 -m scripts.trainCurriculum --procs 32 --trainAllParalell --allSimultaneous --seed 1214 --model AllPara
srun -c 1 -v python3 -m scripts.trainCurriculum --procs 32 --trainAllParalell --allSimultaneous --seed 2330 --model AllPara
srun -c 1 -v python3 -m scripts.trainCurriculum --procs 32 --trainAllParalell --allSimultaneous --seed 8515 --model AllPara
srun -c 1 -v python3 -m scripts.trainCurriculum --procs 32 --trainAllParalell --allSimultaneous --seed 9152 --model AllPara
echo "END run"
