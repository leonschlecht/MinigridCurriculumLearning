#!/bin/bash
#SBATCH --job-name=50krrh_c4
#SBATCH --output=50k_5sRRHcRun_%j_ou2t2.txt
#SBATCH --time=23:59:00
#SBATCH --partition=cpu_normal_stud,cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --verbose
echo "----------Starting RRH--------"
srun -c 1 -v python3 -m scripts.trainCurriculum --procs 32 --iterPerEnv 50000 --model rrh_50k_5step --stepsPerCurric 5 --trainRandomRH --seed 9152
echo "END run"
