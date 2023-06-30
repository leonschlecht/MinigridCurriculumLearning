#!/bin/bash
#SBATCH --job-name=c100_3_3_3TEST
#SBATCH --output=debug_c1_GAc100_3_3_3c1coreun_%j_out.txt
#SBATCH --time=00:14:00
#SBATCH --partition=cpu_normal_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 1 -v python3 -m scripts.trainCurriculum --procs 128 --numCurric 3 --stepsPerCurric 3 --nGen 3 --iterPerEnv 100000 --model debug_100k_3step_3gen_3curric --seed 952399
echo "---------- Cluster Job End ---------------------"
