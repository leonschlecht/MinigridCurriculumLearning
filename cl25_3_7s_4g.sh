#!/bin/bash
#SBATCH --job-name=c_25_7_4g_3
#SBATCH --output=c_25_7_g4_3Run_%j_out.txt
#SBATCH --time=20:59:00
#SBATCH --partition=cpu_long_stud,cpu_normal_stud,cadpool_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=3
#SBATCH --mem=20G
#SBATCH --verbose

echo "Start1"

srun -c 3 -v python3 -m scripts.trainCurriculum --procs 48 --numCurric 3 --stepsPerCurric 7 --nGen 4 --iterPerEnv 25000 --model new_25k_7step_4gen_3curric

echo "END run"
