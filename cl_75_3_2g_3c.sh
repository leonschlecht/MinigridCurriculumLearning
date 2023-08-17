#!/bin/bash
#SBATCH --job-name=c75_32g3
#SBATCH --output=GA_75_32g3_%j_out.txt
#SBATCH --time=47:59:00
#SBATCH --partition=cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 trainCurriculum.py --procs 24 --numCurric 3 --stepsPerCurric 3 --nGen 2 --iterPerEnv 75000 --model 75k_3step_2gen_3curric --seed 9152
echo "---------- Cluster Job End ---------------------"


