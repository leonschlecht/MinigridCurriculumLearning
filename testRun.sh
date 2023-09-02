#!/bin/bash
#SBATCH --job-name=TestRun
#SBATCH --output=new3Test_%j_out.txt
#SBATCH --time=00:15:00
#SBATCH --partition=cadpool_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 2 -v python3 trainCurriculum.py --procs 16 --numCurric 1 --stepsPerCurric 1 --nGen 1 --iterPerEnv 20000 --model TESTdebug3 --seed 8516 --episodes 1

echo "---------- Cluster Job End ---------------------"

