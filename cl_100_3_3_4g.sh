#!/bin/bash
#SBATCH --job-name=c_100_3_4g_3
#SBATCH --output=pt2100k_4gencurricRun_%j_out.txt
#SBATCH --time=21:59:00
#SBATCH --partition=cpu_normal_stud,cpu_long_stud,cadpool_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --verbose

echo "Start "

srun -c 2 -v python3 -m scripts.trainCurriculum --procs 48 --numCurric 3 --stepsPerCurric 3 --nGen 4 --iterPerEnv 100000 --model test12x12_100k_3step_4gen_3curric

echo "END run"