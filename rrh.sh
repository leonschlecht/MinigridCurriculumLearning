#!/bin/bash
#SBATCH --job-name=rrh_c4
#SBATCH --output=r22rh13_cRun_%j_ou2t2.txt
#SBATCH --time=23:59:00
#SBATCH --partition=cpu_normal_stud,cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=3
#SBATCH --mem=10G
#SBATCH --verbose

echo "----------Starting RRH--------"

srun -c 3 -v python3 -m scripts.trainCurriculum --procs 48 --iterPerEnv 50000 --model rrh_50k_5step --stepsPerCurric 5 --trainEpochs 200 --trainRandomRH --seed 3053


echo "END run"
