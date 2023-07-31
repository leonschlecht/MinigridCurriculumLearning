#!/bin/bash
#SBATCH --job-name=AllParav2
#SBATCH --output=AllPara_%j_out.txt
#SBATCH --time=23:59:00
#SBATCH --partition=cpu_normal_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 -m scripts.trainCurriculum --procs 16 --trainAllParalell --asCurric --model AllPara_v2 --noRewardShaping --seed 1
echo "---------- Cluster Job End ---------------------"


