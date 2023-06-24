#!/bin/bash
#SBATCH --job-name=c100_4_3_3
#SBATCH --output=DEL_100_4s_3g_3_%j_out.txt
#SBATCH --time=00:59:00
#SBATCH --partition=cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=6
#SBATCH --mem=30G
#SBATCH --verbose

echo "------------Cluster Job Start-----------------------"

srun -c 9 -v python3 -m scripts.trainCurriculum --procs 64 --numCurric 3 --stepsPerCurric 4 --nGen 3 --iterPerEnv 100000 --model DEBUG_123132 --seed 999

echo "---------- Cluster Job End ---------------------"
