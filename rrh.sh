#!/bin/bash
#SBATCH --job-name=rrh_c3
#SBATCH --output=rrh10_cRun_%j_out.txt
#SBATCH --time=21:59:00
#SBATCH --partition=cpu_normal_stud,cpu_long_stud,cadpool_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=3
#SBATCH --mem=20G
#SBATCH --verbose

echo "Start2"

srun -c 3 -v python3 -m scripts.trainCurriculum --procs 48 --iterPerEnv 150000 --model rrh_3053 --trainEpochs 37 --seed 3053 --trainRandomRH


echo "END run"
