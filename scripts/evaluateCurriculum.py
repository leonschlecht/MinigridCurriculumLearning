import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.Result import Result
from utils import storage
from utils.curriculumHelper import *
from matplotlib.ticker import MaxNLocator


def plotPerformance(allYValues: list[list[float]], allXValues: list[list[int]], maxReward: int, title: str, modelNames: list[str], limitX=False):
    minX = 0
    maxXlist = ([max(x) for x in allXValues])
    print(maxXlist)
    maxXlist.remove(max(maxXlist))
    print("maxX", maxXlist)
    fig, ax = plt.subplots()
    colors = ['blue', 'red', 'green', 'purple']
    linestyles = ['-', '--', '-.', ':']
    new_list = set(maxXlist)
    new_list.remove(max(new_list))
    print(max(new_list))
    for i in range(len(allYValues)):
        ax.plot(allXValues[i], allYValues[i], label=modelNames[i])
        # TODO add scatter again ??
    maxX = max(maxXlist)
    ax.set_ylim([0, maxReward])
    ax.set_xlim([minX, maxX])
    ax.set_xlabel('iterations')
    ax.set_ylabel('reward')
    ax.set_title(title)
    ax.legend()

    mostIterationsDoneXValues = allXValues[0]
    for i in range(len(maxXlist)):
        if maxX == allXValues[i][-1]:
            mostIterationsDoneXValues = allXValues[i]
            break
    # TODO maybe use 250k steps isntead for ticks
    new_tick_locations = [(2 + i * 2) * mostIterationsDoneXValues[1] for i in range(len(mostIterationsDoneXValues) // 2)]
    ax.set_xticks(new_tick_locations)
    ax.set_xticklabels([str(tick // 1000) + 'k' for tick in new_tick_locations])
    plt.show()


def plotSnapshotPerformance(results: list[Result], title: str, modelNamesList):
    y = [res.snapShotScores for res in results]
    x = [[i * res.iterationsPerEnv for i in range(res.epochsTrained)] for res in results]
    maxReward = 1
    plotPerformance(y, x, maxReward, title, modelNamesList, limitX=True)


def plotEpochAvgCurricReward(results: list[Result], title: str, modelNamesList):
    y = [res.avgEpochRewards for res in results]  # TODO NORMALIZE ??
    x = [[i * res.iterationsPerEnv for i in range(res.epochsTrained)] for res in results]
    maxReward = 1
    plotPerformance(y, x, maxReward, title, modelNamesList, limitX=True)


def plotBestCurriculumResults(results: list[Result], title: str, modelNamesList):
    y = [res.bestCurricScore for res in results]
    x = [[i * res.iterationsPerEnv for i in range(res.epochsTrained)] for res in results]

    maxReward = 1

    plotPerformance(y, x, maxReward, title, modelNamesList, limitX=True)


def plotDistributionOfBestCurric(resultClassesList: list[Result], titleInfo: str):
    bestCurricDistr = [[res.bestCurriculaEnvDistribution, res.modelName] for res in resultClassesList]
    plotEnvsUsedDistrSubplot(bestCurricDistr, titleInfo, limitY=True)


def plotSnapshotEnvDistribution(resultClassesList: list[Result], titleInfo: str):
    """
    Plots the distributions of the envs used for 1st step of curricula
    :param resultClassesList:
    :param titleInfo:
    :param modelNamesList:
    :return:
    """
    snapshotDistributions = [[res.snapshotEnvDistribution, res.modelName] for res in resultClassesList]
    plotEnvsUsedDistrSubplot(snapshotDistributions, titleInfo, limitY=True)


def plotDistributionOfAllCurric(resultClassesList: list[Result], titleInfo: str):
    bestCurricDistr = [[res.allCurricDistribution, res.modelName] for res in resultClassesList]
    plotEnvsUsedDistrSubplot(bestCurricDistr, titleInfo)


def plotEnvsUsedDistrSubplot(smallAndLargeDistributions: list[list], titleInfo: str, limitY=False):
    num_subplots = 2
    fig, axes = plt.subplots(nrows=1, ncols=num_subplots, figsize=(8, 5))
    fig.subplots_adjust(bottom=.3)
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
        plotEnvsUsedDistribution(envDistribution, axes[envDistIndex], modelNames, fig, limitY)
    fig.suptitle(titleInfo, fontsize=16)

    plt.show()


def plotEnvsUsedDistribution(allEnvDistributions: list[dict], ax, modelNames, fig, limitY=False):
    num_distributions = len(allEnvDistributions)
    bar_width = 0.5 / num_distributions
    x_offset = -bar_width * (num_distributions - 1) / 2
    maxO = []
    for distrIndex in range(num_distributions):
        envDistribution = allEnvDistributions[distrIndex]
        finalDict = envDistribution
        envs = list(finalDict.keys())
        envOccurrences = list(finalDict.values())
        maxO.append(max(envOccurrences))
        x = np.arange(len(envs))
        x = [xi + x_offset + distrIndex * bar_width for xi in x]
        ax.bar(x, envOccurrences, width=bar_width, label=modelNames[distrIndex])
    ax.set_ylabel('Occurrence')  # TODO only for index == 0?
    ax.set_xticks(np.arange(len(envs)))
    ax.set_xticklabels(envs)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim([0, np.average(maxO) + 1])  # TODO find better way


if __name__ == "__main__":
    evalDirectory = storage.getLogFilePath(["storage", "save", "evaluate"])
    logFilePaths = []
    evalDirectories = next(os.walk(evalDirectory))[1]
    for model in evalDirectories:
        logFilePaths.append(evalDirectory + os.sep + model + os.sep + "status.json")
    parser = argparse.ArgumentParser()
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

    # plotSnapshotEnvDistribution(resultClasses, "Distribution of envs of 1st RH step")
    # plotDistributionOfBestCurric(resultClasses, "Distribution of env occurrence from best performing curricula")
    # plotDistributionOfAllCurric(resultClasses, "Occurence of all curricula of all epochs and generations")
    # plotSnapshotPerformance(resultClasses, "Snapshot Performance", modelNamesList)
    # plotBestCurriculumResults(resultClasses, "Best Curriculum Results", modelNamesList)
    plotEpochAvgCurricReward(resultClasses, "Average Curriculum Reward of all Generations in an epoch", modelNamesList)
    # TODO this should not have a shared x-axis; or at least still use epochs and not scale
