#!/bin/bash
#SBATCH --job-name=c100_4s3g2c
#SBATCH --output=GA_4s3g2c_%j_out.txt
#SBATCH --time=47:59:00
#SBATCH --partition=cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 -m scripts.trainCurriculum --procs 24 --noRewardShaping --numCurric 2 --stepsPerCurric 4 --nGen 3 --iterPerEnv 100000 --model 100k_4step_3gen_2curric_nRS --seed 9152
echo "---------- Cluster Job End ---------------------"


