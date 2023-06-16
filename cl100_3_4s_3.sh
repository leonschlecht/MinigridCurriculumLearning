#!/bin/bash
#SBATCH --job-name=c100_4_3_3
#SBATCH --output=100_4s_3_3_r3_%j_out.txt
#SBATCH --time=47:59:00
#SBATCH --partition=cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --verbose

echo "Start1"

srun -c 4 -v python3 -m scripts.trainCurriculum --procs 64 --numCurric 3 --stepsPerCurric 4 --nGen 3 --iterPerEnv 100000 --model 100k_4step_3gen_3curric_s2330 --seed 2330

echo "END run"

