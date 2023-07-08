#!/bin/bash
#SBATCH --job-name=allPara2
#SBATCH --output=allPara2_%j.txt
#SBATCH --time=23:59:00
#SBATCH --partition=cpu_normal_stud,cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --verbose
echo "Start2"
srun -c 1 -v python3 -m scripts.trainCurriculum --procs 32 --trainAllParalell --seed 8515 --model AllPara
srun -c 1 -v python3 -m scripts.trainCurriculum --procs 32 --trainAllParalell --seed 9152 --model AllPara
echo "END run"
