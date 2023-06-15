#!/bin/bash
#SBATCH --job-name=c100_4_3_3
#SBATCH --output=pt3curricRun_%j_out.txt
#SBATCH --time=20:59:00
#SBATCH --partition=cpu_normal_stud,cpu_long_stud,cadpool_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=3
#SBATCH --mem=20G
#SBATCH --verbose

echo "Start1"

srun -c 3 -v python3 -m scripts.trainCurriculum --procs 48 --numCurric 3 --stepsPerCurric 4 --nGen 3 --iterPerEnv 100000 --model test12x12_100k_4step_3gen_3curric
srun -c 3 -v python3 -m scripts.trainCurriculum --procs 48 --numCurric 3 --stepsPerCurric 4 --nGen 3 --iterPerEnv 100000 --model s9152_100k_4step_3gen_3curric --seed 9152

echo "END run"
