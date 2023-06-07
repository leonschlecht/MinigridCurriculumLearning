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
    snapshotDistributions = [res.snapshotEnvDistribution for res in resultClassesList]
    largeSize = []
    smallSize = []
    for s in snapshotDistributions:
        isSmallSize = True
        for key in s.keys():
            if "16x16" in key:
                isSmallSize = False
                break
        if isSmallSize:
            smallSize.append(s)
        else:
            largeSize.append(s)
    # plotEnvsUsedDistribution(largeSize, titleInfo + " large", modelNamesList)
    plotEnvsUsedDistrSubplot(snapshotDistributions, titleInfo, modelNamesList)
    exit()


# TODO allCurricDsitribution

def plotEnvsUsedDistrSubplot(smallAndLargeDistributions: list[dict], titleInfo: str, modelNamesList):
    num_subplots = 2
    fig, axes = plt.subplots(nrows=1, ncols=num_subplots, figsize=(10, 10))
    smallDistributions = []
    largeDistributions = []
    # Prepare the 2 subplot lists
    for distr in smallAndLargeDistributions:
        numericStrDict = {numKey.split('-')[-1]: val for numKey, val in distr.items()}
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
                keyFound = True
        if not keyFound:
            smallDistributions.append(finalDict)

    assert len(smallDistributions) + len(largeDistributions) == len(smallAndLargeDistributions)
    assert len(smallAndLargeDistributions) > len(smallDistributions) > 0
    assert len(largeDistributions) > 0
    for envDistIndex in range(num_subplots):
        if envDistIndex == 0:
            envDistribution = smallDistributions
        elif envDistIndex == 1:
            envDistribution = largeDistributions
        else:
            raise Exception("Invalid env distribution index")
        print("plotting w", envDistribution)
        plotEnvsUsedDistribution(envDistribution, titleInfo, modelNamesList, axes[envDistIndex])
    plt.show()

def plotEnvsUsedDistribution(allEnvDistributions: list[dict], titleInfo: str, modelNamesList, ax):
    num_distributions = len(allEnvDistributions)
    bar_width = 0.5 / num_distributions
    print(allEnvDistributions)
    x_offset = -bar_width * (num_distributions - 1) / 2
    for distrIndex in range(num_distributions):
        envDistribution = allEnvDistributions[distrIndex]
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

        ax.bar(x, envOccurrences, width=bar_width, label="123")

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

    modelNames = [res.modelName for res in resultClasses]

    plotSnapshotEnvDistribution(resultClasses, "Distribution of envs of best performing curricula", modelNames)
    plotDistributionOfBestCurric(resultClasses, "Distribution of env occurrence from best performing curricula", modelNames)
    # TODO all envs distr plot

    plotSnapshotPerformance(resultClasses, "Snapshot Performance", modelNames)
    plotBestCurriculumResults(resultClasses, "Best Curriculum Results", modelNames)
    plotEpochAvgCurricReward(resultClasses, "Average Curriculum Reward of all Generations in an epoch",
                             modelNames)  # TODO this should not have a shared x-axis; or at least still use epochs and not scale

    # TODO experiment comparison: long iterPerStep with many Gen, with low iterPerStep with low gen
    # TODO plot showing differences between earlier and later generations
    # TODO plot the snapshot vs curricReward problem ---> do some experiments. How does the curricLength influence results? How does gamma influence results?

    # TODO: low prio stuff
    # find out a way to properly plot the difficulty list / maybe how it influences the results; and maybe how you can improve it so that it is not even needed in the first place
    # save the plots
