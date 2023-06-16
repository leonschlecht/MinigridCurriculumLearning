#!/bin/bash
#SBATCH --job-name=c5_50_3_3_3
#SBATCH --output=c5_50_3_3_3curricRun_%j_out.txt
#SBATCH --time=20:59:00
#SBATCH --partition=cpu_normal_stud,cpu_long_stud,cadpool_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --verbose

echo "Start2"

srun -c 2 -v python3 -m scripts.trainCurriculum --procs 48 --numCurric 3 --stepsPerCurric 3 --nGen 3 --iterPerEnv 50000 --model s8515_50k_3step_3gen_3curric --trainEpochs 100 --seed 8515

echo "END run"
