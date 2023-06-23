seeds = [1,
         # 9152,
         # 2330,
         # 1214,
         8515]
experiments = [
    "NSGA_100k_3s_3g_3c",
    "NSGA_100k_3s_4g_3c",
    "NSGA_100k_4s_3g_3c",

]

template = '''#!/bin/bash
#SBATCH --job-name=c_{jobName}
#SBATCH --output=c_{name}_%j_out.txt
#SBATCH --time=23:59:00
#SBATCH --partition=cpu_normal_stud,cpu_long_stud
#SBATCH --exclude=cp2019-11,cc1l01
#SBATCH --cpus-per-task=3
#SBATCH --mem=20G
#SBATCH --verbose
echo "------------Cluster Job Start-----------------------"
srun -c 3 -v python3 -m scripts.trainCurriculum --procs 48 --numCurric {numCurric} --stepsPerCurric {stepsPerCurric] --nGen {gen} --iterPerEnv {iterPerEnv} --model {modelName} --seed {seed}
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
            print(iterations)
            fullScripts.append(filename)
            with open(filename, "w") as file:
                file.write(template.format(
                    jobName=fullExperimentName,
                    outputName=experiment,
                    numCurric=curricAmount,
                    stepPerCurric=stepsPerCurric,
                    gen=nGen,
                    iterPerEnv=iterations,
                    modelName=experiment,
                    seed=seed)
                )

        def executeScripts():
            print("executing...")
            print(fullScripts)

        if __name__ == "__main__":
            createScripts()
            executeScripts()
