#!/bin/bash
#SBATCH --job-name=rrh_c4
#SBATCH --output=250kRRHcRun_%j_ou2t2.txt
#SBATCH --time=23:59:00
#SBATCH --partition=cpu_normal_stud,cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --verbose
echo "----------Starting RRH--------"
srun -c 1 -v python3 -m scripts.trainCurriculum --procs 32 --iterPerEnv 250000 --model rrh_250k_3step --stepsPerCurric 3 --trainRandomRH --seed 9152
echo "END run"
