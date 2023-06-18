#!/bin/bash
#SBATCH --job-name=allPara2
#SBATCH --output=allPara2_%j.txt
#SBATCH --time=11:59:00
#SBATCH --partition=cpu_normal_stud,cpu_long_stud,cadpool_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --verbose

echo "Start2"

srun -c 2 -v python3 -m scripts.trainCurriculum --procs 32 --trainAllParalell --seed 9152 --model AllPara_Seed9152
srun -c 2 -v python3 -m scripts.trainCurriculum --procs 32 --trainAllParalell --seed 2330 --model AllPara_Seed2330

echo "END run"
