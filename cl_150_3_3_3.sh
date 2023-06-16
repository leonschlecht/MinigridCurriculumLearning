#!/bin/bash
#SBATCH --job-name=c5_150_3_3_3
#SBATCH --output=c5_150_3_3_3_cRun_%j_out.txt
#SBATCH --time=20:59:00
#SBATCH --partition=cpu_normal_stud,cpu_long_stud,cadpool_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=3
#SBATCH --mem=20G
#SBATCH --verbose

echo "Start2"

srun -c 3 -v python3 -m scripts.trainCurriculum --procs 48 --numCurric 3 --stepsPerCurric 3 --nGen 3 --iterPerEnv 150000 --model s2529_150k_3step_3gen_3curric --trainEpochs 35 --seed 2529

echo "END run"
