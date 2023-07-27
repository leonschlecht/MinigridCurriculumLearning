import argparse
import os

import matplotlib.pyplot as plt
import pandas
import pandas as pd
import seaborn as sns

from scripts.Result import Result
from utils import storage
from utils.curriculumHelper import *
from matplotlib.ticker import MaxNLocator
import numpy as np

OFFSET = 10000


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
                                "group": result.iterationsPerEnv,
                                iterationSteps: result.iterationsList[i]})  #

        # Create DF2 that contains the distributions etc. (iterationNr column does not make sense here)
        tmp = result.snapshotEnvDistribution.keys()
        snapshotHelper = [[] for _ in tmp]
        allCurricHelper = [[] for _ in tmp]
        bestCurricHelper = [[] for _ in tmp]
        for k in tmp:
            if "6x6" in k:
                snapshotHelper[0] = (result.snapshotEnvDistribution[k])
                allCurricHelper[0] = (result.allCurricDistribution[k])
                bestCurricHelper[0] = (result.bestCurriculaEnvDistribution[k])
            elif "8x8" in k:
                snapshotHelper[1] = (result.snapshotEnvDistribution[k])
                allCurricHelper[1] = (result.allCurricDistribution[k])
                bestCurricHelper[1] = (result.bestCurriculaEnvDistribution[k])
            elif "10x10" in k:
                snapshotHelper[2] = (result.snapshotEnvDistribution[k])
                allCurricHelper[2] = (result.allCurricDistribution[k])
                bestCurricHelper[2] = (result.bestCurriculaEnvDistribution[k])
            else:
                snapshotHelper[3] = (result.snapshotEnvDistribution[k])
                allCurricHelper[3] = (result.allCurricDistribution[k])
                bestCurricHelper[3] = (result.bestCurriculaEnvDistribution[k])
        distributionHelper.append({
            "6x6s": snapshotHelper[0],
            "8x8s": snapshotHelper[1],
            "10x10s": snapshotHelper[2],
            "12x12s": snapshotHelper[3],
            "6x6c": bestCurricHelper[0],
            "8x8c": bestCurricHelper[1],
            "10x10c": bestCurricHelper[2],
            "12x12c": bestCurricHelper[3],
            "6x6a": allCurricHelper[0],
            "8x8a": allCurricHelper[1],
            "10x10a": allCurricHelper[2],
            "12x12a": allCurricHelper[3],
            seedKey: result.seed,
            "group": result.iterationsPerEnv,
            iterationSteps: result.iterationsList[0],
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


def filterDf(filters: list[str], dataFrame, models, showCanceled=False):
    """
    Given a list of models and the main dataframe, it filters all the relevant id columns matching the @val prefix
    :param filters:
    :param rheaFilter:
    :param showCanceled:
    :param dataFrame: the main dataframe
    :param models: a list of all model names (unique values in id column of the df)
    :return:
    """
    filteredDf = []
    for m in models:
        append = True
        for filterOption in filters:
            if filterOption not in m or \
                    (filterOption == "50k" and "150k" in m) or \
                    (filterOption == "50k" and "250k" in m) or \
                    (filterOption == "GA" and "NSGA" in m):  # TODO find way not having to do this manually every time
                append = False
                break
        if append:
            if "C_" in m and not showCanceled:
                continue
            filteredDf.append(dataFrame[dataFrame["id"] == m])

    return filteredDf


def getUserInputForMultipleComparisons(models: list, comparisons: int, scoreDf, distrDf) -> tuple:
    modelsEntered: int = 0
    usedModels = []
    filteredScoreDfList = []
    filteredDistrDfList = []  # TODO
    for i in range(len(models)):
        print(f"{i}: {models[i]}")
    print("filter options: NSGA, GA, RndRH, allParalell. \n\t iter[number]", args.filter)
    if args.filter:
        filters = []
        if args.rhea:
            filters.append("GA_")  # slightly hacky to avoid the "GA" == duplicate check above (since NSGA is also in GA string)
            if args.rrh:
                filters.append("RRH")  # todo ???
        if args.ga:
            filters.append("GA")
        if args.nsga:
            filters.append("NSGA")
        if args.rrhOnly:
            filters.append("RndRH")
        if args.iter != 0:
            filters.append(str(args.iter) + "k")
        if args.steps:
            # get all NSGA, GA and RRH runs (or make differnetaion here too ?)
            # TODO
            pass
        if args.gen:
            # get ALL NSGA or GA runs
            # plot them
            # TODO
            print()
        if args.curric:
            # get all NSGA, GA, RRH runs
            # TODO
            pass
        filteredScoreDfList = filterDf(filters, scoreDf, models, showCanceled=args.showCanceled)
        filteredDistrDfList = filterDf(filters, distrDf, models, showCanceled=args.showCanceled)
    else:
        while modelsEntered < comparisons:
            val = (input(f"Enter model number ({modelsEntered}/{comparisons}): "))
            if val.isdigit() and int(val) < len(models) and val not in usedModels:
                modelsEntered += 1
                usedModels.append(val)
                filteredScoreDfList.append(scoreDf[scoreDf["id"] == models[int(val)]])
                filteredDistrDfList.append(distrDf[distrDf["id"] == models[int(val)]])
            else:
                if val == "RndRH" or val == "NSGA" or val == "GA" or val == "allParalell":
                    val = [val]
                    filteredScoreDfList = filterDf(val, scoreDf, models)
                    filteredDistrDfList = filterDf(val, distrDf, models)
                    break
                print("Model doesnt exist or was chosen already. Enter again")
    print("Models entered. Beginning visualization process")
    print(filteredDistrDfList)
    return filteredScoreDfList, filteredDistrDfList


def plotMultipleLineplots(filteredDfList):
    fig, ax = plt.subplots(figsize=(12, 8))  # Increase figure size
    sns.set_theme(style="darkgrid")
    for df in filteredDfList:
        sns.lineplot(x=iterationSteps, y="snapshotScore", data=df, label=df.head(1)["id"].item(), ax=ax,
                     errorbar=args.errorbar)  # Now you can plot each 'DataFrame' group separately
    # TODO somehow make other variant accessible too where you need grouping
    # df = pd.concat([df.assign(DataFrame=i) for i, df in enumerate(filteredDfList)])

    # group_col = "group" if "group" in df.columns else "DataFrame"
    # grouped = df.groupby(group_col)
    #
    # for name, group in grouped:
    #     #sns.lineplot(x='iterationSteps', y='snapshotScore', data=group, label=str(name), ax=ax)

    ax.set_ylabel("evaluation reward")
    ax.set_xlabel("iterations")

    box = ax.get_position()
    # Edit this line out to move the legend out of the plot
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_xlim((0, args.xIterations))
    ax.set_ylim((0, 1))
    plt.title("mean performance")
    plt.show()


def removeExperimentPrefix(dictDf):
    """
    Removes the "GA_" prefix in the id column (or whichever variant was used like NSGA or RRH)
    :param dictDf:
    :return:
    """
    split = dictDf["id"].values
    experiment = split[0]
    if "GA" in experiment and not "NSGA" in experiment:
        filteredWord = "GA"
    elif "NSGA" in experiment:
        filteredWord = "NSGA"
    elif "RndRH" in experiment:
        filteredWord = "RndRH"
    elif "allParalell" in experiment:
        return experiment
    else:
        raise Exception("Something went wrong with the experiment name")
    words = experiment.split("_")

    if filteredWord == "GA" or filteredWord == "NSGA" or filteredWord == "RndRH":
        iterations = words[1] + "_"
        steps = words[2].split("tep")[0] + "_"  # cut "3step" to 3s
    else:  # TODO probably not needed
        iterations = ""
        steps = ""

    if filteredWord == "GA" or filteredWord == "NSGA":
        gen = words[3].split("en")[0] + "_"  # cut "3gen" to 3g
    else:
        gen = ""

    if filteredWord == "GA" or filteredWord == "NSGA":
        curric = words[4].split("urric")[0]  # cut "3curric" to 3c
    elif filteredWord == "RndRH":
        curric = words[3].split("urric")[0]
        assert curric != ""
    else:
        curric = ""
    return iterations + steps + gen + curric


def showDistrVisualization(aggregatedDf, columnsToVisualize):
    fig, ax = plt.subplots(figsize=(12, 8))

    group_col = "group" if 'group' in aggregatedDf.columns else 'DataFrame'
    aggregatedDf['sort_col'] = aggregatedDf['id'].str.split('_', n=1, expand=True)[0].str.replace('k', '').astype('int')
    aggregatedDf = aggregatedDf.sort_values(by=['sort_col', group_col])
    aggregatedDf = aggregatedDf.drop('sort_col', axis=1)

    grouped_df = aggregatedDf[columnsToVisualize].groupby('id').agg(['mean', 'std'])
    grouped_df = grouped_df.reset_index()
    melted_df = grouped_df.melt(id_vars='id', var_name=['Column', 'Statistic'], value_name='Value')

    # Sort and convert to categorical
    melted_df['id'] = pd.Categorical(melted_df['id'], categories=aggregatedDf['id'].unique(), ordered=True)

    sns.barplot(data=melted_df, x='id', y='Value', hue='Column', errorbar=args.errorbar)
    plt.ylabel('Value')
    plt.title("Environment Distribution")
    plt.ylabel('distribution')
    plt.xlabel('')
    plt.xticks(rotation=-45, ha='left', fontsize=9)
    plt.subplots_adjust(bottom=0.2)
    if args.normalize:
        plt.ylim((0, .55))
    plt.show()

def includeNormalizedColumns(aggregatedDf, prefix):
    """
    Updates the df to include the normalized columns of the environment distributions
    :param aggregatedDf: DataFrame to update
    :param prefix: prefix to use for column names
    :return: updated DataFrame
    """
    envSizes = ['6x6', '8x8', '10x10', '12x12']
    normalizedDistributions = {size: [] for size in envSizes}

    for i, row in aggregatedDf.iterrows():
        totalEnvs = sum(row[f"{size}{prefix}"] for size in envSizes)
        for size in envSizes:
            normalizedDistributions[size].append(1.0 * row[f"{size}{prefix}"] / totalEnvs)

    for size in envSizes:
        column_name = f"{size}n"
        aggregatedDf[column_name] = normalizedDistributions[size]

    return aggregatedDf


def showTrainingTimePlot(aggregatedDf):
    fig, ax = plt.subplots(figsize=(12, 8))
    group_col = "group" if 'group' in aggregatedDf.columns else 'DataFrame'  # TODO probably not for every experiment
    if args.rhea:
        aggregatedDf['sort_col'] = aggregatedDf['id'].str.split('_', n=1, expand=True)[0].str.replace('k', '').astype('int')
        aggregatedDf = aggregatedDf.sort_values(by=['sort_col', group_col])
        aggregatedDf = aggregatedDf.drop('sort_col', axis=1)
        sns.barplot(x='id', y='sumTrainingTime', hue=group_col, dodge=False, data=aggregatedDf, ax=ax)
    else:
        sns.barplot(x='id', y='sumTrainingTime', data=aggregatedDf, ax=ax)

    plt.ylabel('training time (hours)')
    plt.xlabel('')
    title = "Training Time"
    if args.title:
        title = args.title
    plt.title(title)
    plt.xticks(rotation=-45, ha='left', fontsize=9)
    plt.subplots_adjust(bottom=0.2)

    # TODO the legend part might be specific to some of the settings and not universal
    if args.rhea:
        legend = ax.legend()
        labels = [int(item.get_text()) for item in legend.get_texts()]
        labels = [f"{label // 1000}k steps" for label in labels]
        for label, text in zip(legend.texts, labels):
            label.set_text(text)
        labels = [item.get_text() for item in ax.get_xticklabels()]
        for i in range(len(labels)):
            labelI = labels[i]
            labelI = "_".join(labelI.split("_")[1:])
            labels[i] = labelI
        ax.set_xticklabels(labels)
    ax.set_ylim((0, 96))
    plt.show()


def plotAggregatedBarplot(filteredDfList):
    assert len(filteredDfList) > 0, "filteredDfList empty"
    for f in filteredDfList:
        for fullModelName in f["id"]:
            expParams = fullModelName.split("_")
            noModelName = "_".join(expParams[1:])
            break
        f["id"] = noModelName + "_" + expParams[0]
    aggregatedDf = pd.concat([df.assign(DataFrame=i) for i, df in enumerate(filteredDfList)])

    if args.trainingTime:
        showTrainingTimePlot(aggregatedDf)

    column_prefixes = {'snapshotDistr': 's', 'curricDistr': 'c', 'allDistr': 'a'}
    for arg, prefix in column_prefixes.items():
        if getattr(args, arg):
            if args.normalize:
                aggregatedDf = includeNormalizedColumns(aggregatedDf, prefix)
                columns_to_visualize = [f'6x6n', f'8x8n', f'10x10n', f'12x12n', 'id']
            else:
                columns_to_visualize = [f'6x6{prefix}', f'8x8{prefix}', f'10x10{prefix}', f'12x12{prefix}', 'id']
            showDistrVisualization(aggregatedDf, columns_to_visualize)

    print("---- Done ----")


def main(comparisons: int):
    pathList = ["storage", "_evaluate"]
    useCrossoverMutationPath = args.crossoverMutation
    if useCrossoverMutationPath:
        pathList.append("SOBOL_GA_75_3_3_3")
    evalDirBasePath = storage.getLogFilePath(pathList)
    fullLogfilePaths = []
    evalDirectories = next(os.walk(evalDirBasePath))[1]
    statusJson = "status.json"
    specificModelList = []
    for model in evalDirectories:
        if model == "old" or (not args.crossoverMutation and "SOBOL" in model):
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
    scoreDf = scoreDf[scoreDf[iterationSteps] < args.xIterations + OFFSET]
    models = scoreDf["id"].unique()
    sns.set_theme(style="darkgrid")
    print("------------------\n\n\n")

    if args.model is None and not args.skip:
        filteredScoreDf, filteredDistrDf = getUserInputForMultipleComparisons(models, comparisons, scoreDf, distrDf)
        if args.plotScores:  # TODO args option for args.useScore <= snapshotScores, ucrricScores, idk what else there was
            plotMultipleLineplots(filteredScoreDf)
        plotAggregatedBarplot(filteredDistrDf)

    if args.model is not None and not args.skip:
        filteredScoreDf = scoreDf[scoreDf["id"] == args.model]
        sns.lineplot(x=iterationSteps, y="snapshotScore", data=filteredScoreDf, label=args.model, errorbar=args.errorbar)
        plt.show()
    if args.skip:
        print("starting evaluation. . .")
        for m in models:
            if "C_" in m and not args.showCanceled:
                continue
            modelDf = scoreDf[scoreDf["id"] == m]
            filteredIterDf = modelDf[iterationSteps]
            try:
                firstIterVal = filteredIterDf[0]
            except:
                continue
            occur = len(filteredIterDf[filteredIterDf == firstIterVal])
            print(f"{occur} experiments done with {m}")
            modelDf = modelDf[modelDf[iterationSteps] < args.xIterations + OFFSET]
            sns.lineplot(x=iterationSteps, y="snapshotScore", data=modelDf, label=m, errorbar=args.errorbar)
            plt.xlabel('Index..')
            plt.ylabel('training time (hours)')
            plt.title(args.title)
            plt.legend()
            plt.show()

    # filepath = Path('./out.csv') # TODO DF save as csv
    # filepath.parent.mkdir(parents=True, exist_ok=True)
    # scoreDf.to_csv(filepath, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Option to select a single model for evaluation")
    parser.add_argument("--title", default=None, type=str, help="Title of the distribution plots")
    parser.add_argument("--comparisons", default=2, help="Choose how many models you want to compare")
    parser.add_argument("--skip", action="store_true", default=False, help="Debug option to skip the UI part and see each model 1 by 1")
    parser.add_argument("--crossoverMutation", action="store_true", default=False, help="Select the crossovermutation varied experiments")

    parser.add_argument("--trainingTime", action="store_true", default=False, help="Show training time plots")
    parser.add_argument("--snapshotDistr", action="store_true", default=False, help="Show first step distribution plots")
    parser.add_argument("--curricDistr", action="store_true", default=False, help="show all best curricula distributions plots")
    parser.add_argument("--allDistr", action="store_true", default=False, help="Show all distribution plots")
    parser.add_argument("--plotScores", action="store_true", default=False, help="Plots the score")
    parser.add_argument("--normalize", action="store_true", default=False, help="Whether or not to normlaize the env distributions")

    parser.add_argument("--iter", default=0, type=int, help="filter for iterations")
    parser.add_argument("--xIterations", default=1100000, type=int, help="#of iterations to show on the xaxis")
    parser.add_argument("--steps", action="store_true", default=False, help="filter for #curricSteps")
    parser.add_argument("--gen", action="store_true", default=False, help="Whether to filter #gen")
    parser.add_argument("--curric", action="store_true", default=False, help="whether to filter for #curricula")
    parser.add_argument("--rhea", action="store_true", default=False, help="Only using rhea runs")
    parser.add_argument("--nsga", action="store_true", default=False, help="Only using GA runs")
    parser.add_argument("--ga", action="store_true", default=False, help="Only using NSGA runs")
    parser.add_argument("--rrh", action="store_true", default=False, help="Include RRH runs, even if --rhea was speicifed")
    parser.add_argument("--rrhOnly", action="store_true", default=False, help="Only RRH runs")
    parser.add_argument("--showCanceled", action="store_true", default=False, help="Whether to use canceled runs too")
    parser.add_argument("--errorbar", default=None, type=str, help="What type of errorbar to show on the lineplots. (Such as sd, ci etc)")
    args = parser.parse_args()
    args.filter = args.comparisons == 2 and (
            args.crossoverMutation or args.iter or args.steps or args.gen or args.curric or args.rrhOnly or args.rhea or args.nsga or args.ga)
    main(int(args.comparisons))
