import os
import subprocess

seeds = [1,
         9152,
         2330,
         1214,
         8515]
experiments = []


def make_experiments(algo: str, iterationsPerEnv: int, stepsPerCurric: int, nGen: int, numCurric: int):
    """

    :param algo:
    :param iterationsPerEnv: iterations per Env in 100k
    :param stepsPerCurric:
    :param nGen:
    :param numCurric:
    :return:
    """
    experiment = f"{algo}_{iterationsPerEnv}k_{stepsPerCurric}s_{nGen}g_{numCurric}c"
    if experiment in experiments:
        raise Exception("Experiment with those parameters already exists!")
    experiments.append(experiment)


template = '''#!/bin/bash
#SBATCH --job-name=c{jobName}
#SBATCH --output={outputName}_%j_out.txt
#SBATCH --time=47:59:00
#SBATCH --partition=cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 4 -v python3 -m scripts.trainCurriculum --procs 32 --numCurric {numCurric} --stepsPerCurric {stepsPerCurric} --nGen {gen} --iterPerEnv {iterPerEnv} --model {modelName} --seed {seed}
echo "---------- Cluster Job End ---------------------"
'''

fullScripts = []


# TODO probably should add an args option to decide which evol algo to use

def getDigits(string):
    numeric_value = ""
    for char in string:
        if char.isdigit():
            numeric_value += char
    return int(numeric_value)


def createScripts():
    for seed in seeds:
        for experiment in experiments:
            fullExperimentName = experiment + "_s_" + str(seed)
            filename = f"{fullExperimentName}.sh"
            expSplit = experiment.split("_")
            algo = (expSplit[0])
            iterations = getDigits(expSplit[1]) * 1000
            stepsPerCurric = getDigits(expSplit[2])
            nGen = getDigits(expSplit[3])
            curricAmount = getDigits(expSplit[4])
            fullScripts.append(filename)
            if not os.path.isfile(filename):
                with open(filename, "w+") as file:
                    formatted = template.format(
                        jobName=fullExperimentName,
                        outputName=experiment,
                        numCurric=curricAmount,
                        stepsPerCurric=stepsPerCurric,
                        gen=nGen,
                        iterPerEnv=iterations,
                        modelName=experiment,
                        seed=seed)
                    file.write(formatted)
            else:
                print("file exists already", filename)
        print("All Scripts created!")


def executeScripts():
    for script in fullScripts:
        path = "./" + script
        print("path", path)
        result = subprocess.call(["sbatch", path])

        if result == 0:
            print("Script executed successfully!")
        else:
            raise Exception("Script execution failed.")
    print("All Scripts started")


if __name__ == "__main__":
    # make_experiments(algo="NSGA", iterationsPerEnv=100, stepsPerCurric=3, nGen=3, numCurric=3)  # EXPERIMENT DONE
    make_experiments(algo="NSGA", iterationsPerEnv=50, stepsPerCurric=3, nGen=3, numCurric=3) # cancel tmrw
    make_experiments(algo="NSGA", iterationsPerEnv=25, stepsPerCurric=4, nGen=3, numCurric=3) # cancel tmrw
    make_experiments(algo="NSGA", iterationsPerEnv=25, stepsPerCurric=3, nGen=4, numCurric=3) # cancel tmrw
    make_experiments(algo="NSGA", iterationsPerEnv=25, stepsPerCurric=3, nGen=3, numCurric=3) # cancel tomorrow
    make_experiments(algo="NSGA", iterationsPerEnv=25, stepsPerCurric=3, nGen=3, numCurric=3) # cancel tomorrow
    make_experiments(algo="NSGA", iterationsPerEnv=25, stepsPerCurric=7, nGen=1, numCurric=3) # Runs 48h on multiple cpu

    # make_experiments(algo="NSGA", iterationsPerEnv=100, stepsPerCurric=3, nGen=4, numCurric=3) # NOT STARTED
    # make_experiments(algo="NSGA", iterationsPerEnv=100, stepsPerCurric=4, nGen=3, numCurric=3) # NOT STARTED

    createScripts()

    # executeScripts()
