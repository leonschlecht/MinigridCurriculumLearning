#!/bin/bash
#SBATCH --job-name=c75_3_3_3
#SBATCH --output=c5_75_3_3_3curricRun_%j_out.txt
#SBATCH --time=20:59:00
#SBATCH --partition=cpu_normal_stud,cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --verbose

echo "------------Cluster Job Start-----------------------"

srun -c 2 -v python3 -m scripts.trainCurriculum --procs 32 --numCurric 3 --stepsPerCurric 3 --nGen 3 --iterPerEnv 75000 --model 75k_3step_3gen_3curric_s8258 --trainEpochs 67 --seed 8258


echo "---------- Cluster Job End ---------------------"

