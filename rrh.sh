#!/bin/bash
#SBATCH --job-name=rrh
#SBATCH --output=rrh_150_%j_out.txt
#SBATCH --time=23:59:00
#SBATCH --partition=cpu_normal_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 -m scripts.trainCurriculum --procs 24 --iterPerEnv 250000 --model rrh_250k_3step --stepsPerCurric 3 --trainRandomRH --seed 2529

echo "---------- Cluster Job End ---------------------"
