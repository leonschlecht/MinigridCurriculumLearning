#!/bin/bash
#SBATCH --job-name=rrh
#SBATCH --output=rrh_%j_out.txt
#SBATCH --time=71:59:00
#SBATCH --partition=cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 trainCurriculum.py --noRewardShaping --procs 24 --iterPerEnv 100000 --model rrh_100k_6c_3gen --stepsPerCurric 3 --numCurric 6 --trainRandomRH --seed 9152
echo "---------- Cluster Job End ---------------------"
