#!/bin/bash
#SBATCH --job-name=rrh_c
#SBATCH --output=rrh_cRun_%j_out.txt
#SBATCH --time=20:59:00
#SBATCH --partition=cpu_normal_stud,cpu_long_stud,cadpool_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --verbose

echo "Start2"

srun -c 3 -v python3 -m scripts.trainCurriculum --procs 48 --iterPerEnv 150000 --model s_1_rrh --trainEpochs 37 --trainRandomRH
srun -c 3 -v python3 -m scripts.trainCurriculum --procs 48 --iterPerEnv 150000 --model s8515_rrh --trainEpochs 37 --seed 8515 --trainRandomRH


echo "END run"
