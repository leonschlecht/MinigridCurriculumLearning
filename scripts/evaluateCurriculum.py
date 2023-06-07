import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.Result import Result
from utils import storage
from utils.curriculumHelper import *
from matplotlib.ticker import MaxNLocator


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
    new_tick_locations = [(2 + i * 2) * mostIteratiosnDoneXValues[1] for i in
                          range(len(mostIteratiosnDoneXValues) // 2)]  # TODO maybe use 250k steps isntead
    ax.set_xticks(new_tick_locations)
    ax.set_xticklabels([str(tick // 1000) + 'k' for tick in new_tick_locations])
    plt.show()


def plotSnapshotPerformance(results: list[Result], title: str, modelNamesList):
    y = [res.snapShotScores for res in results]
    x = [[i * res.iterationsPerEnv for i in range(res.epochsTrained)] for res in results]
    maxReward = 1
    plotPerformance(y, x, maxReward, title, modelNamesList)


def plotEpochAvgCurricReward(results: list[Result], title: str, modelNamesList):
    y = [res.avgEpochRewards for res in results]  # TODO NORMALIZE ??
    x = [[i * res.iterationsPerEnv for i in range(res.epochsTrained)] for res in results]
    maxReward = 1
    plotPerformance(y, x, maxReward, title, modelNamesList)


def plotBestCurriculumResults(results: list[Result], title: str, modelNamesList):
    y = [res.bestCurricScore for res in results]
    x = [[i * res.iterationsPerEnv for i in range(res.epochsTrained)] for res in results]

    maxReward = 1

    plotPerformance(y, x, maxReward, title, modelNamesList)


def plotDistributionOfBestCurric(resultClassesList: list[Result], titleInfo: str, modelNamesList):
    bestCurricDistr = [res.bestCurriculaEnvDistribution for res in resultClassesList]
    plotEnvsUsedDistribution(bestCurricDistr, titleInfo, modelNamesList)


def plotSnapshotEnvDistribution(resultClassesList: list[Result], titleInfo: str, modelNamesList: list):
    """
    Plots the distributions of the envs used for 1st step of curricula
    :param resultClassesList:
    :param titleInfo:
    :param modelNamesList:
    :return:
    """
    snapshotDistributions = [[res.snapshotEnvDistribution, res.modelName] for res in resultClassesList]

    # plotEnvsUsedDistribution(largeSize, titleInfo + " large", modelNamesList)
    plotEnvsUsedDistrSubplot(snapshotDistributions, titleInfo)
    exit()


# TODO allCurricDsitribution

def plotEnvsUsedDistrSubplot(smallAndLargeDistributions: list[dict], titleInfo: str):
    num_subplots = 2
    fig, axes = plt.subplots(nrows=1, ncols=num_subplots, figsize=(10, 10))
    smallDistributions = []
    largeDistributions = []
    modelNamesLarge = []
    modelNamesSmall = []
    # Prepare the 2 subplot lists
    for distr in smallAndLargeDistributions:
        numericStrDict = {numKey.split('-')[-1]: val for numKey, val in distr[0].items()}
        keyValues = sorted([int(fullKey.split("x")[0]) for fullKey in numericStrDict])
        numStrKeyMapping = []
        for numKey in keyValues:
            for dictKey in numericStrDict:
                if str(numKey) == dictKey.split('x')[0]:
                    numStrKeyMapping.append((dictKey, numKey))
                    break

        sortedKeys = [fullKey for fullKey, _ in numStrKeyMapping]
        finalDict = {k: numericStrDict[k] for k in sortedKeys}
        keyFound = False
        for key in finalDict.keys():
            if "16x16" in key:
                largeDistributions.append(finalDict)
                modelNamesLarge.append(distr[1])
                keyFound = True
        if not keyFound:
            smallDistributions.append(finalDict)
            modelNamesSmall.append(distr[1])

    assert len(smallDistributions) + len(largeDistributions) == len(smallAndLargeDistributions)
    assert len(smallAndLargeDistributions) > len(smallDistributions) > 0
    assert len(largeDistributions) > 0
    for envDistIndex in range(num_subplots):
        if envDistIndex == 0:
            envDistribution = smallDistributions
            modelNames = modelNamesSmall
        elif envDistIndex == 1:
            envDistribution = largeDistributions
            modelNames = modelNamesLarge
        else:
            raise Exception("Invalid env distribution index")
        print("plotting w", envDistribution)
        plotEnvsUsedDistribution(envDistribution, titleInfo, axes[envDistIndex], modelNames)
    plt.show()

def plotEnvsUsedDistribution(allEnvDistributions: list[dict], titleInfo: str, ax, modelNames):
    num_distributions = len(allEnvDistributions)
    bar_width = 0.5 / num_distributions
    print(allEnvDistributions)
    x_offset = -bar_width * (num_distributions - 1) / 2
    for distrIndex in range(num_distributions):
        envDistribution = allEnvDistributions[distrIndex]
        # name = envDistribution["name"]
        print(envDistribution)
        # Create Mapping from the string MiniGrid-DoorKey-6x6-
        # to the actual numbers. Results in tuples of ('6x6', 6) and so on for sorting later
        shortenedEnvFreqMapping = {numKey.split('-')[-1]: val for numKey, val in envDistribution.items()}
        levelSizeAsNumbers: list[int] = sorted([int(fullKey.split("x")[0]) for fullKey in shortenedEnvFreqMapping])
        numStrKeyMapping = []
        for numKey in levelSizeAsNumbers:
            for dictKey in shortenedEnvFreqMapping:
                if str(numKey) == dictKey.split('x')[0]:
                    numStrKeyMapping.append((dictKey, numKey))
                    break

        sortedKeys = [fullKey for fullKey, _ in numStrKeyMapping]
        finalDict = {k: shortenedEnvFreqMapping[k] for k in sortedKeys}

        envs = list(finalDict.keys())
        envOccurrences = list(finalDict.values())

        x = np.arange(len(envs))
        x = [xi + x_offset + distrIndex * bar_width for xi in x]

        ax.bar(x, envOccurrences, width=bar_width, label=modelNames[distrIndex])

    ax.set_ylabel('Occurrence')
    ax.set_title(titleInfo)
    ax.set_xticks(np.arange(len(envs)))
    ax.set_xticklabels(envs)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))  # move legend outside of plot
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))



if __name__ == "__main__":
    evalDirectory = storage.getLogFilePath(["storage", "save", "evaluate"])
    logFilePaths = []
    evalDirectories = next(os.walk(evalDirectory))[1]
    for model in evalDirectories:
        logFilePaths.append(evalDirectory + os.sep + model + os.sep + "status.json")
    parser = argparse.ArgumentParser()
    # General parameters
    parser.add_argument("--model", default=None, help="Option to select a single model for evaluation")
    args = parser.parse_args()

    resultClasses = []
    for logFilePath in logFilePaths:
        modelName = Path(logFilePath).parent.name
        if args.model == modelName:
            with open(logFilePath, 'r') as f:
                trainingInfoDictionary = json.loads(f.read())
            assert trainingInfoDictionary is not None
            resultClasses = [Result(trainingInfoDictionary, modelName, logFilePath)]
            break
        if os.path.exists(logFilePath):
            with open(logFilePath, 'r') as f:
                trainingInfoDictionary = json.loads(f.read())
            assert trainingInfoDictionary is not None
            resultClasses.append(Result(trainingInfoDictionary, modelName, logFilePath))
        else:
            print(f"Path '{logFilePath}' doesnt exist!")
            # raise Exception(f"Path '{logFilePath}' doesnt exist!")

    modelNamesList = [res.modelName for res in resultClasses]

    plotSnapshotEnvDistribution(resultClasses, "Distribution of envs of best performing curricula", modelNamesList)
    plotDistributionOfBestCurric(resultClasses, "Distribution of env occurrence from best performing curricula", modelNamesList)
    # TODO all envs distr plot

    plotSnapshotPerformance(resultClasses, "Snapshot Performance", modelNamesList)
    plotBestCurriculumResults(resultClasses, "Best Curriculum Results", modelNamesList)
    plotEpochAvgCurricReward(resultClasses, "Average Curriculum Reward of all Generations in an epoch",
                             modelNamesList)  # TODO this should not have a shared x-axis; or at least still use epochs and not scale

    # TODO experiment comparison: long iterPerStep with many Gen, with low iterPerStep with low gen
    # TODO plot showing differences between earlier and later generations
    # TODO plot the snapshot vs curricReward problem ---> do some experiments. How does the curricLength influence results? How does gamma influence results?

    # TODO: low prio stuff
    # find out a way to properly plot the difficulty list / maybe how it influences the results; and maybe how you can improve it so that it is not even needed in the first place
    # save the plots
