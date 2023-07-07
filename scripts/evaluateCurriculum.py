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
import numpy as np


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


iterationSteps = "iterationSteps"


def getSpecificModel(specificModelList: list, modelName: str):
    assert specificModelList != [], f"Model List must not be empty. Modelname {modelName}"
    results = []
    for logPath in specificModelList:
        with open(logPath, 'r') as f:
            trainingInfoDictionary = json.loads(f.read())
        assert trainingInfoDictionary is not None
        if (trainingInfoDictionary[epochsDone]) > 1:
            results.append(Result(trainingInfoDictionary, modelName, logPath))
        else:
            print("Epochs <= 1", logPath)

    scoreHelper = []
    distributionHelper = []
    medianLen = []
    for result in results:
        medianLen.append(result.epochsTrained)
        # Create DF1 for scores etc
        for i in range(len(result.snapShotScores)):
            scoreHelper.append({"snapshotScore": result.snapShotScores[i],
                                "bestCurricScore": result.bestCurricScore[i],
                                "avgEpochRewards": result.avgEpochRewards[i],
                                "id": modelName,
                                iterationSteps: result.iterationsList[i]})

        distributionHelper.append({"snapshotDistribution": result.snapshotEnvDistribution,
                                   "bestCurricDistribution": result.bestCurriculaEnvDistribution,
                                   "allCurricDistribution": result.allCurricDistribution,
                                   "seed": result.seed,
                                   sumTrainingTime: result.trainingTimeSum,
                                   "id": modelName})
    rewardScoreDf = pd.DataFrame(scoreHelper)
    medianLen = int(np.median(medianLen)) + 1  # TODO just make suer all experiments are done to full so its not needed
    rewardScoreDf = rewardScoreDf[rewardScoreDf[iterationSteps] <= results[0].iterationsPerEnv * medianLen]

    # TODO assert all iterPerEnv are equal ?
    distributionDf = pd.DataFrame(distributionHelper)
    return rewardScoreDf, distributionDf


def getAllModels(logfilePaths: list[list]):
    scoreDf = pd.DataFrame()
    distrDf = pd.DataFrame()
    for logfilePath in logfilePaths:
        tmpScoreDf, tmpDistrDf = getSpecificModel(logfilePath[0], logfilePath[1])
        # scoreDf = scoreDf.append(tmpScoreDf)
        scoreDf = pd.concat([scoreDf, tmpScoreDf], ignore_index=True)
        # distrDf = distrDf.append(tmpDistrDf)
        distrDf = pd.concat([distrDf, tmpDistrDf], ignore_index=True)
    return scoreDf, distrDf


def filterDf(val, scoreDf, models):
    """
    Given a list of models and the main dataframe, it filters all the relevant id columns matching the @val prefix
    :param val: the prefix to be filtered. E.g. "RndRH"
    :param scoreDf: the main dataframe
    :param models: a list of all model names (unique values in id column of the df)
    :return:
    """
    filteredDf = []
    for m in models:
        if val in m and "C_" not in m:
            if val == "GA" and "NSGA" in m:
                continue
            filteredDf.append(scoreDf[scoreDf["id"] == m])
    return filteredDf


def getUserInputForMultipleComparisons(models: list, comparisons: int, scoreDf, distrDf):
    modelsEntered: int = 0
    usedModels = []
    filteredScoreDf = []
    filteredDistrDf = []  # TODO
    for i in range(len(models)):
        print(f"{i}: {models[i]}")
    print("filter options: NSGA, GA, RndRH, allParalell")
    val = None
    while modelsEntered < comparisons:
        val = (input(f"Enter model number ({modelsEntered}/{comparisons}): "))
        if val.isdigit() and int(val) < len(models) and val not in usedModels:
            modelsEntered += 1
            usedModels.append(val)
            filteredScoreDf.append(scoreDf[scoreDf["id"] == models[int(val)]])
            filteredDistrDf.append(distrDf[distrDf["id"] == models[int(val)]])
        else:
            if val == "RndRH" or val == "NSGA" or val == "GA" or val == "allParalell":
                filteredScoreDf = filterDf(val, scoreDf, models)
                filteredDistrDf = filterDf(val, distrDf, models)
                break
            print("Model doesnt exist or was chosen already. Enter again")
    print("Models entered. Beginning visualization process")
    return filteredScoreDf, filteredDistrDf, val


def plotMultipleLineplots(filteredDf):
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    for df in filteredDf:
        sns.lineplot(x=iterationSteps, y="snapshotScore", data=df, label=df.head(1)["id"].item(), ax=ax)
    ax.set_ylabel("evaluation reward .")
    ax.set_xlabel("iterations .")
    # ax.set_ylim(bottom=0.6)
    plt.tight_layout()  # Add this line to adjust the layout and prevent legend cutoff
    # plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.legend()
    plt.show()


def plotAggrgatedBarplot(filteredDf: list[pd.DataFrame], filterWord):
    assert len(filteredDf) > 0, "filteredDf empty"
    if filterWord is not None:
        for dictDf in filteredDf:
            split = dictDf["id"].values
            experiment = split[0]
            print(experiment)
            words = experiment.split("_")
            steps = words[2].split("tep")[0]
            gen = words[3].split("en")[0]
            curric = words[4].split("urric")[0]
            newId = words[1] + "_" + steps + "_" + gen + "_" + curric
            dictDf["id"] = newId

    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    aggregatedDf = pd.DataFrame()
    for df in filteredDf:
        aggregatedDf = pd.concat([aggregatedDf, df], ignore_index=True)

    sns.barplot(x="id", y=sumTrainingTime, data=aggregatedDf, ax=ax)
    plt.ylabel('training time (hours)')
    plt.title(f'Training time for {filterWord}')
    plt.show()


def main(comparisons: int):
    evalDirBasePath = storage.getLogFilePath(["storage", "_evaluate"])
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
        scoreDf, distrDf = getSpecificModel(specificModelList, args.model)
    else:
        scoreDf, distrDf = getAllModels(fullLogfilePaths)

    models = scoreDf["id"].unique()
    sns.set_theme(style="dark")
    # TODO ask for comparison nrs if not given by --comparisons
    print("------------------\n\n\n")
    if args.model is None and not args.skip:
        filteredScoreDf, filteredDistrDf, filterWord = getUserInputForMultipleComparisons(models, comparisons, scoreDf, distrDf)
        # plotMultipleLineplots(filteredDf)
        plotAggrgatedBarplot(filteredDistrDf, filterWord)

    if args.model is not None and not args.skip:
        filteredScoreDf = scoreDf[scoreDf["id"] == args.model]
        sns.lineplot(x=iterationSteps, y="snapshotScore", data=filteredScoreDf, label=args.model)
        plt.show()
    if args.skip:
        print("starting evaluation. . .")
        for m in models:
            if "C_" in m and not args.showCanceled:
                continue
            modelDf = scoreDf[scoreDf["id"] == m]
            filteredIterDf = modelDf[iterationSteps]
            firstIterVal = filteredIterDf[0]
            occur = len(filteredIterDf[filteredIterDf == firstIterVal])
            print(f"{occur} experiments done with {m}")
            sns.lineplot(x=iterationSteps, y="snapshotScore", data=modelDf, label=m)
            plt.xlabel('Index')  # Replace 'Index' with the appropriate x-axis label
            plt.ylabel('sumTrainingTime')  # Replace 'sumTrainingTime' with the appropriate y-axis label
            plt.title('Barplot of sumTrainingTime')  # Replace 'Barplot of sumTrainingTime' with the appropriate title for your plot
            plt.legend()
            plt.show()

    # filepath = Path('./out.csv') # TODO DF save as csv
    # filepath.parent.mkdir(parents=True, exist_ok=True)
    # scoreDf.to_csv(filepath, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Option to select a single model for evaluation")
    parser.add_argument("--comparisons", default=2, help="Choose how many models you want to compare")
    parser.add_argument("--skip", action="store_true", default=False, help="Debug option to skip the UI part and see each model 1 by 1")
    parser.add_argument("--showCanceled", action="store_true", default=False, help="Debug option to skip the UI part and see each model 1 by 1")
    args = parser.parse_args()
    main(int(args.comparisons))
