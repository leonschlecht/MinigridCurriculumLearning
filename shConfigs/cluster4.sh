#!/bin/bash
#SBATCH --job-name=curriculumTest
#SBATCH --output=pt2curricRun_%j_out.txt
#SBATCH --time=11:59:00
#SBATCH --partition=cpu_short_stud,cadpool_stud,cpu_normal_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --verbose

echo "Start "

srun -c 2 -v python3 -m scripts.trainCurriculum --procs 48 --numCurric 3 --stepsPerCurric 3 --nGen 3 --iterPerEnv 75000 --model test12x12_75k_3step_3gen_3curric

echo "END run"
