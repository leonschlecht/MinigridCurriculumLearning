#!/bin/bash
#SBATCH --job-name=50krrh_c4
#SBATCH --output=50k_5sRRHcRun_%j_ou2t2.txt
#SBATCH --time=23:59:00
#SBATCH --partition=cpu_normal_stud,cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --verbose
echo "----------Starting RRH--------"
srun -c 1 -v python3 -m scripts.trainCurriculum --procs 32 --iterPerEnv 250000 --model rrh_250k_3step --stepsPerCurric 3 --trainRandomRH --seed 2529
srun -c 1 -v python3 -m scripts.trainCurriculum --procs 32 --iterPerEnv 250000 --model rrh_250k_3step --stepsPerCurric 3 --trainRandomRH --seed 8258
srun -c 1 -v python3 -m scripts.trainCurriculum --procs 32 --iterPerEnv 250000 --model rrh_250k_3step --stepsPerCurric 3 --trainRandomRH --seed 1517
srun -c 1 -v python3 -m scripts.trainCurriculum --procs 32 --iterPerEnv 250000 --model rrh_250k_3step --stepsPerCurric 3 --trainRandomRH --seed 185
srun -c 1 -v python3 -m scripts.trainCurriculum --procs 32 --iterPerEnv 250000 --model rrh_250k_3step --stepsPerCurric 3 --trainRandomRH --seed 3053
echo "END run"
