#!/bin/bash
#SBATCH --job-name=dyn
#SBATCH --output=GA_dyn_%j_out.txt
#SBATCH --time=71:59:00
#SBATCH --partition=cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 trainCurriculum.py --procs 24 --noRewardShaping --stepsPerCurric 3 --nGen 2 --numCurric 4 --iterPerEnv 100000 --model 5_100k_3step_2gen_4curric --dynamicObstacle --seed 2330
echo "---------- Cluster Job End ---------------------"


