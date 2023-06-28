#!/bin/bash
#SBATCH --job-name=allPara2
#SBATCH --output=allPara2_%j.txt
#SBATCH --time=11:59:00
#SBATCH --partition=cpu_normal_stud,cpu_long_stud,cadpool_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=9
#SBATCH --mem=24G
#SBATCH --verbose
echo "Start2"
srun -c 3 -v python3 -m scripts.trainCurriculum --procs 32 --trainAllParalell --seed 1 --model AllPara
srun -c 3 -v python3 -m scripts.trainCurriculum --procs 32 --trainAllParalell --seed 1214 --model AllPara
srun -c 3 -v python3 -m scripts.trainCurriculum --procs 32 --trainAllParalell --seed 2330 --model AllPara
echo "END run"
