#!/bin/bash
#SBATCH --job-name=curriculumTest
#SBATCH --output=curricRun_%j_out.txt
#SBATCH --time=11:59:00
#SBATCH --partition=cpu_short_stud,cadpool_stud,cpu_normal_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --verbose

echo "Start "

srun -c 2 -v python -m scripts.trainCurriculum --procs 32 --numCurric 3 --stepsPerCurric 3 --nGen 3 --iterPerEnv 100000 --model 12x12_100k_3step_3gen_3curric

echo "END run"