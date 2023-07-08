#!/bin/bash
#SBATCH --job-name=v3GA75_4c2g
#SBATCH --output=v3GA_75_4c2gcurricRun_%j_out.txt
#SBATCH --time=47:59:00
#SBATCH --partition=cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 1 -v python3 -m scripts.trainCurriculum --procs 32 --numCurric 4 --stepsPerCurric 3 --nGen 2 --iterPerEnv 75000 --model 75k_3step_2gen_4curric --useNSGA --seed 1
echo "---------- Cluster Job End ---------------------"

