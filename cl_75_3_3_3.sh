#!/bin/bash
#SBATCH --job-name=c_75_3_3_3
#SBATCH --output=c_75_3_3_3curricRun_%j_out.txt
#SBATCH --time=20:59:00
#SBATCH --partition=cpu_normal_stud,cpu_long_stud,cadpool_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=3
#SBATCH --mem=20G
#SBATCH --verbose

echo "Start"

srun -c 3 -v python3 -m scripts.trainCurriculum --procs 48 --numCurric 3 --stepsPerCurric 3 --nGen 3 --iterPerEnv 75000 --model s2330_75k_3step_3gen_3curric --trainEpochs 67 --seed 2330
srun -c 3 -v python3 -m scripts.trainCurriculum --procs 48 --numCurric 3 --stepsPerCurric 3 --nGen 3 --iterPerEnv 75000 --model s1214_75k_3step_3gen_3curric --trainEpochs 67 --seed 1214
srun -c 3 -v python3 -m scripts.trainCurriculum --procs 48 --numCurric 3 --stepsPerCurric 3 --nGen 3 --iterPerEnv 75000 --model s8515_75k_3step_3gen_3curric --trainEpochs 67 --seed 8515

echo "END run"
