import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scripts.Result import Result
from utils import storage
from utils.curriculumHelper import *
from matplotlib.ticker import MaxNLocator


def plotPerformance(allYValues: list[list[float]], allXValues: list[list[int]], maxYValue: int, title: str,
                    modelNames: list[str], limitX=False):
    fig, ax = plt.subplots()

    minX = 0
    maxXlist = ([max(x) for x in allXValues])
    if len(maxXlist) > 1:
        maxXlist.remove(max(maxXlist))
    # colors = ['blue', 'red', 'green', 'purple']
    # linestyles = ['-', '--', '-.', ':']
    new_list = set(maxXlist)
    if len(new_list) > 1:
        new_list.remove(max(new_list))
    for j in range(len(allYValues)):
        ax.plot(allXValues[j], allYValues[j], label=modelNames[j])
        # TODO add scatter again ??
    maxX = max(maxXlist)
    ax.set_ylim([0, maxYValue])
    ax.set_xlim([minX, maxX])
    ax.set_xlabel('iterations')
    ax.set_ylabel('reward')
    ax.set_title(title)
    ax.legend()

    mostIterationsDoneXValues = allXValues[0]
    for j in range(len(maxXlist)):
        if maxX == allXValues[j][-1]:
            mostIterationsDoneXValues = allXValues[j]
            break
    # TODO maybe use 250k (or so) steps isntead for ticks
    new_tick_locations = [(2 + j * 2) * mostIterationsDoneXValues[1] for j in
                          range(len(mostIterationsDoneXValues) // 2)]
    ax.set_xticks(new_tick_locations)
    ax.set_xticklabels([str(tick // 1000) + 'k' for tick in new_tick_locations])
    plt.show()


def plotSnapshotPerformance(results: list[Result], title: str, modelNamesList):
    y = [res.snapShotScores for res in results]
    x = [[i * res.iterationsPerEnv for i in range(res.epochsTrained)] for res in results]
    maxReward = 1

    plotPerformance(y, x, maxReward, title, modelNamesList, limitX=True)


def plotDifficulty(results: list[Result], title: str, modelNamesList):
    y = [res.difficultyList for res in results]
    x = [[i * res.iterationsPerEnv for i in range(res.epochsTrained + 1)] for res in results]
    maxYValue = 2
    # TODO fix y-axis in this case so it shows integers only
    plotPerformance(y, x, maxYValue, title, modelNamesList, limitX=True)


def plotEpochAvgCurricReward(results: list[Result], title: str, modelNamesList):
    y = [res.avgEpochRewards for res in results]  # TODO NORMALIZE ?? (already done i think ??)
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
    numSubplots = 2
    fig, axes = plt.subplots(nrows=1, ncols=numSubplots, figsize=(8, 5))
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
    assert len(smallAndLargeDistributions) >= len(smallDistributions) >= 0
    assert len(largeDistributions) >= 0

    distributionsList = []
    if len(smallDistributions) > 0:
        distributionsList.append([smallDistributions, modelNamesSmall])
    if len(largeDistributions) > 0:
        distributionsList.append([largeDistributions, modelNamesLarge])

    envDistrIndex = 0
    for distribution in distributionsList:
        if len(distributionsList) == 1:
            axesObj = axes
        else:
            axesObj = axes[envDistrIndex]
        plotEnvsUsedDistribution(distribution[0], axesObj, distribution[1], fig, limitY)
        envDistrIndex += 1
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
    ax.set_ylim([0, np.average(maxO) + 1])  # TODO find better way to cut things off


def getSpecificModel(specificModelList: list, modelName: str):
    assert specificModelList != [], "Model List must not be empty"

    results = []
    for logPath in specificModelList:
        with open(logPath, 'r') as f:
            trainingInfoDictionary = json.loads(f.read())
        assert trainingInfoDictionary is not None
        if (trainingInfoDictionary[epochsDone]) > 1:
            results.append(Result(trainingInfoDictionary, modelName, logPath))
        else:
            print("Epochs <= 1")

    dfRow = []
    for result in results:
        dfRow.append({"snapshotScore": result.snapShotScores,
                      "bestCurricScore": result.bestCurricScore,  # TODO careful with AllPara
                      "avgEpochRewards": result.avgEpochRewards,  # TODO careful with allPara
                      "snapshotDistribution": result.snapshotEnvDistribution,
                      "bestCurricDistribution": result.bestCurriculaEnvDistribution,
                      "allCurricDistribution": result.allCurricDistribution,
                      "id": modelName,
                      "iterPerEnv": result.iterationsPerEnv})
    df = pd.DataFrame(dfRow)
    # TODO assert all iterPerEnv are equal ?
    return df


def getAllModels(logfilePaths: list[list]):
    df = pd.DataFrame()
    for logfilePath in logfilePaths:
        tmpDf: pd.DataFrame = getSpecificModel(logfilePath[0], logfilePath[1])
        df = df.append(tmpDf)
    return df


def main():
    evalDirBasePath = storage.getLogFilePath(["storage", "save", "evaluate"])
    fullLogfilePaths = []
    evalDirectories = next(os.walk(evalDirBasePath))[1]
    statusJson = "status.json"
    specificModelList = []
    for model in evalDirectories:
        if model == "old":
            continue
        path = evalDirBasePath + os.sep + model + os.sep
        json_files = [f for f in os.listdir(path) if f == statusJson]
        fullLogfilePaths.append([[], model])
        for jsonFile in json_files:
            fullLogfilePaths[-1][0].append(path + jsonFile)
        seededExperimentsDirs = (next(os.walk(path)))[1]
        for seededExperiment in seededExperimentsDirs:
            path = evalDirBasePath + os.sep + model + os.sep + seededExperiment + os.sep
            jsonFIlesHelper = [f for f in os.listdir(path) if f == statusJson]
            for jsonFile2 in jsonFIlesHelper:
                fullLogfilePaths[-1][0].append(path + jsonFile2)
        if model == args.model:
            specificModelList = fullLogfilePaths[-1][0]
            break

    if args.model is not None:
        dataFrame = getSpecificModel(specificModelList, args.model)
    else:
        dataFrame = getAllModels(fullLogfilePaths)

    sns.set_theme(style="darkgrid")
    sns.lineplot(x="snapshotScore", y="iterPerEnv", data=dataFrame)
    plt.show()
    exit()

    """
    # DEPRECATED 
    modelNamesList = [res.modelName for res in resultClasses]

    plotSnapshotPerformance(resultClasses, "First Step Performance per Epoch", modelNamesList)
    # plotDifficulty(resultClasses, "Overview of Difficulty List", modelNamesList)
    plotSnapshotEnvDistribution(resultClasses, "First Step Env Distribution")

    plotBestCurriculumResults(resultClasses, "Reward of Best Curriculum per Epoch", modelNamesList)
    plotDistributionOfBestCurric(resultClasses, "Best Curricula Env Distribution")

    plotEpochAvgCurricReward(resultClasses, "Average Curriculum Reward of all Generations in an epoch", modelNamesList)
    plotDistributionOfAllCurric(resultClasses, "Occurence of all curricula of all epochs and generations")
    """
    # TODO the plotDistributionOfAllCurric should not have a shared x-axis; or at least still use epochs and not scale ???


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Option to select a single model for evaluation")
    args = parser.parse_args()
    main()
