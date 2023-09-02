#!/bin/bash
#SBATCH --job-name=testc100nRS
#SBATCH --output=testGA_100nRS_%j_out.txt
#SBATCH --time=00:10:00
#SBATCH --partition=cadpool_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 ../trainCurriculum.py --procs 16 --noRewardShaping --numCurric 1 --stepsPerCurric 1 --nGen 1 --iterPerEnv -1 --model de2_100k_3step_3gen_3curric --dynamicObstacle --seed 9152
echo "---------- Cluster Job End ---------------------"


