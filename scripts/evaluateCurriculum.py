import os
from pathlib import Path

import matplotlib.pyplot as plt

from scripts.Result import Result
from utils import storage
from utils.curriculumHelper import *


def plotPerformance(allYValues: list[list[float]], allXValues: list[list[int]], maxReward: int, title: str, modelNames: list[str]):
    minX = 0
    maxX = max([max(x) for x in allXValues]) * 1.05
    fig, ax = plt.subplots()
    colors = ['blue', 'red', 'green', 'purple']
    linestyles = ['-', '--', '-.', ':']
    for i in range(len(allYValues)):
        ax.plot(allXValues[i], allYValues[i], label=modelNames[i], color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)])
        # TODO add scatter again ??

    ax.set_ylim([0, maxReward])
    ax.set_xlim([minX, maxX])
    ax.set_xlabel('iterations')
    ax.set_ylabel('reward')
    ax.set_title(title)
    ax.legend()

    mostIteratiosnDoneXValues = max(allXValues, key=lambda x: x[-1])
    new_tick_locations = [(2 + i * 2) * mostIteratiosnDoneXValues[1] for i in range(len(mostIteratiosnDoneXValues) // 2)]  # TODO maybe use 250k steps isntead
    ax.set_xticks(new_tick_locations)
    ax.set_xticklabels([str(tick // 1000) + 'k' for tick in new_tick_locations])
    plt.show()


def plotSnapshotPerformance(results: list[Result], title: str, modelNamesList):
    y = [res.snapShotScores for res in results]
    x = [[i * res.iterationsPerEnv for i in range(res.epochsTrained)] for res in results]
    maxReward = 1
    plotPerformance(y, x, maxReward, title, modelNamesList)


def plotEpochAvgCurricReward(results: list[Result], title: str, modelNamesList):
    y = [res.avgEpochRewards for res in results] # TODO NORMALIZE ??
    x = [[i * res.iterationsPerEnv for i in range(res.epochsTrained)] for res in results]
    maxReward = 1
    plotPerformance(y, x, maxReward, title, modelNamesList)


def plotBestCurriculumResults(results: list[Result], title: str, modelNamesList):
    y = [res.bestCurricScore for res in results]
    x = [[i * res.iterationsPerEnv for i in range(res.epochsTrained)] for res in results]

    maxReward = 1

    plotPerformance(y, x, maxReward, title, modelNamesList)


def plotDistributionOfBestCurric(resultClassesList: list[Result], titleInfo: str, modelNamesList):
    plotEnvsUsedDistribution(resultClassesList[0].bestCurriculaEnvDistribution, titleInfo, modelNamesList)


def plotSnapshotEnvDistribution(resultClassesList: list[Result], titleInfo: str, modelNamesList: list):
    snapshotDistributions = [res.snapshotEnvDistribution for res in resultClassesList]
    plotEnvsUsedDistrSubplot(snapshotDistributions, titleInfo, modelNamesList)


# TODO allCurricDsitribution

def plotEnvsUsedDistrSubplot(allEnvDistributions: list[dict], titleInfo: str, modelNamesList):
    num_subplots = len(allEnvDistributions)
    fig, axes = plt.subplots(nrows=num_subplots, ncols=1, figsize=(12, 12))
    for envDistIndex in range(num_subplots):
        envDistribution = allEnvDistributions[envDistIndex]
        numericStrDict = {numKey.split('-')[-1]: val for numKey, val in envDistribution.items()}
        keyValues = sorted([int(fullKey.split("x")[0]) for fullKey in numericStrDict])

        numStrKeyMapping = []
        for numKey in keyValues:
            for dictKey in numericStrDict:
                if str(numKey) == dictKey.split('x')[0]:
                    numStrKeyMapping.append((dictKey, numKey))
                    break

        sortedKeys = [fullKey for fullKey, _ in numStrKeyMapping]
        finalDict = {k: numericStrDict[k] for k in sortedKeys}

        envs = list(finalDict.keys())
        envOccurrences = list(finalDict.values())

        bar_container = axes[envDistIndex].bar(envs, envOccurrences, label=modelNamesList[envDistIndex])

        # Add labels to the bars
        axes[envDistIndex].bar_label(bar_container, labels=envOccurrences, fontsize=12, padding=5)

        axes[envDistIndex].set_ylabel('Occurrence')
        axes[envDistIndex].set_title(titleInfo)
        axes[envDistIndex].set_ylim(0, max(envOccurrences) * 1.1)
        axes[envDistIndex].legend()

    plt.show()


def plotEnvsUsedDistribution(allEnvDistributions: list[dict], titleInfo: str, modelNamesList):
    # extract the numeric part of the keys of the envDistribution
    print(allEnvDistributions)
    fig, ax = plt.subplots()

    for i in range(len(allEnvDistributions)):
        envDistribution = allEnvDistributions[i]
        numericStrDict = {numKey.split('-')[-1]: val for numKey, val in envDistribution.items()}
        keyValues = sorted([int(fullKey.split("x")[0]) for fullKey in numericStrDict])
        numStrKeyMapping = []
        for numKey in keyValues:
            for dictKey in numericStrDict:
                if str(numKey) == dictKey.split('x')[0]:
                    numStrKeyMapping.append((dictKey, numKey))
                    break

        sortedKeys = [fullKey for fullKey, _ in numStrKeyMapping]
        finalDict = {k: numericStrDict[k] for k in sortedKeys}

        envs = list(finalDict.keys())
        envOccurrences = list(finalDict.values())
        bar_container = ax.bar(envs, envOccurrences, label=modelNamesList[0])

    ax.set_ylabel('Occurrence')
    ax.set_title(titleInfo)
    ax.set_ylim(0, 30)

    # Add labels to the bars
    ax.bar_label(bar_container, labels=envOccurrences, fontsize=12, padding=5)
    ax.legend()
    plt.show()

    # TODO find way to plot multiple models at once (and show some relevant legend for info of model name or sth like that)
    # TODO extra args option to only load 1 model
    # TODO experiment comparison: long iterPerStep with many Gen, with low iterPerStep with low gen

    # TODO plot the snapshot vs curricReward problem ---> do some experiments. How does the curricLength influence results? How does gamma influence results?

    # TODO: low prio stuff
    # find out a way to properly plot the difficulty list / maybe how it influences the results; and maybe how you can improve it so that it is not even needed in the first place
    # save the plots


if __name__ == "__main__":
    evalDirectory = storage.getLogFilePath(["storage", "save", "evaluate"])
    logFilePaths = []
    evalDirectories = next(os.walk(evalDirectory))[1]
    for model in evalDirectories:
        logFilePaths.append(evalDirectory + os.sep + model + os.sep + "status.json")

    resultClasses = []
    for logFilePath in logFilePaths:
        if "16x16" in logFilePath:
            continue
        if os.path.exists(logFilePath):
            with open(logFilePath, 'r') as f:
                trainingInfoDictionary = json.loads(f.read())
            assert trainingInfoDictionary is not None
            modelName = Path(logFilePath).parent.name
            resultClass = Result(trainingInfoDictionary, modelName, logFilePath)
            resultClasses.append(resultClass)
        else:
            raise Exception(f"Path '{logFilePath}' doesnt exist!")

    modelNames = [res.modelName for res in resultClasses]

    plotSnapshotEnvDistribution(resultClasses, "Distribution of envs of best performing curricula", modelNames)
    # plotEnvsUsedDistribution(resultClasses, "all Curric Distribution", modelNames)
    # plotEnvsUsedDistribution(resultClasses, "snapshot Distribution", modelNames)
    # TODO are there other distributions that are useful?

    #plotSnapshotPerformance(resultClasses, "Snapshot Performance", modelNames)
    #plotBestCurriculumResults(resultClasses, "Best Curriculum Results", modelNames)
    # plotEpochAvgCurricReward(resultClasses, "Average Curriculum Reward of all Generations in an epoch", modelNames) # TODO this should not have a shared x-axis; or at least still use epochs and not scale

