import argparse
import os
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from curricula.Result import Result
from utils import storage
from utils.curriculumHelper import *

# find out if i can go back 1 directory and then run the scripts. this would solve the same issue

OFFSET = 1000


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
    scoreHelperList = []
    distributionHelperList = []
    medianLen = []
    splitDistrDf = None
    for result in results:
        medianLen.append(result.epochsTrained)
        for i in range(len(result.snapShotScores)):
            scoreHelperList.append(result.getScoreAtStepI(i))
        distributionHelperList.append(result.getDistributions(isDoorKey))
        splitDistrDf = pd.DataFrame(result.getSplitDistrList(isDoorKey))
    rewardScoreDf = pd.DataFrame(scoreHelperList)
    assert not np.isnan(medianLen).all(), f"medianLen Nan, {medianLen} for modelName {modelName}"
    medianLen = int(np.median(medianLen)) + 1  # TODO just make suer all experiments are done to full so its not needed
    rewardScoreDf = rewardScoreDf[rewardScoreDf[iterationSteps] <= results[0].iterationsPerEnv * medianLen]

    # TODO assert all iterPerEnv are equal ?
    distributionDf = pd.DataFrame(distributionHelperList)
    return rewardScoreDf, distributionDf, splitDistrDf


def getAllDfs(logfilePaths):
    """
    Given the logfilepaths, it returns a df with all the relevant runs.
    This does basic filtering, e.g. if you want DoorKey runs then the dynamic obstalce runs wont be shown
    :param logfilePaths:
    :return:
    """
    scoreDf = pd.DataFrame()
    fullDistrDf = pd.DataFrame()
    splitDistrDf = pd.DataFrame()

    for modelName in logfilePaths:
        jsonPaths = logfilePaths[modelName]
        # if no filter option ? --> then get the list of all
        if "old" in jsonPaths[0]:
            continue
        if args.crossoverMutation and "_c" not in jsonPaths[0] and "_m" not in modelName:
            continue
        if not isDoorKey and "Doorkey" in jsonPaths[0]:
            continue
        elif isDoorKey and "DynamicObstacle" in jsonPaths[0]:
            continue
        # TODO further filters ???
        tmpScoreDf, tmpDistrDf, tmpSplitDf = getSpecificModel(jsonPaths, modelName)
        scoreDf = pd.concat([scoreDf, tmpScoreDf], ignore_index=True)
        fullDistrDf = pd.concat([fullDistrDf, tmpDistrDf], ignore_index=True)
        splitDistrDf = pd.concat([splitDistrDf, tmpSplitDf], ignore_index=True)
    scoreDf = scoreDf[scoreDf[iterationSteps] < args.xIterations + OFFSET]
    assert not scoreDf.empty
    assert not fullDistrDf.empty
    return scoreDf, fullDistrDf, splitDistrDf


def filterDf(filters: list[str], df, models, showCanceled=False):
    """
    Given a list of models and the main dataframe, it filters all the relevant id columns matching the @val prefix
    """

    def passes_filters(colId, filterList):
        for filterOption in filterList:
            if filterOption in colId:
                if filterOption == 'GA' and 'NSGA' in colId or \
                        (filterOption == '50k' and ('150k' in colId or '250k' in colId)) or \
                        (not showCanceled and "C_" in colId):
                    return False
            else:
                return False
        return True

    filteredDf = df[df['id'].apply(lambda x: passes_filters(x, filters))]
    return filteredDf


def getUserInputForMultipleComparisons(models: list, comparisons: int, scoreDf, distrDf, splitDistrDf) -> tuple:
    modelsEntered: int = 0
    usedModels = []
    filteredScoreDf = pd.DataFrame()
    filteredDistrDf = pd.DataFrame()
    filteredSplitDistrDf = pd.DataFrame()
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
        filteredScoreDf = filterDf(filters, scoreDf, models, showCanceled=args.showCanceled)
        filteredDistrDf = filterDf(filters, distrDf, models, showCanceled=args.showCanceled)
        filteredSplitDistrDf = filterDf(filters, splitDistrDf, models, showCanceled=args.showCanceled)
    else:
        for i in range(len(models)):
            print(f"{i}: {models[i]}")
        print("filter options: NSGA, GA, RndRH, allParalell. \n\t iter[number]")
        while modelsEntered < comparisons:
            val = (input(f"Enter model number ({modelsEntered}/{comparisons}): "))
            if val.isdigit() and int(val) < len(models) and val not in usedModels:
                modelsEntered += 1
                usedModels.append(val)
                filteredScoreDf = pd.concat([filteredScoreDf, scoreDf[scoreDf["id"] == models[int(val)]]], ignore_index=True)
                filteredDistrDf = pd.concat([filteredDistrDf, distrDf[distrDf["id"] == models[int(val)]]], ignore_index=True)
                filteredSplitDistrDf = pd.concat([filteredSplitDistrDf, splitDistrDf[splitDistrDf["id"] == models[int(val)]]], ignore_index=True)
            else:
                if val == "RndRH" or val == "NSGA" or val == "GA" or val == "allParalell":
                    val = [val]
                    filteredScoreDf = filterDf(val, scoreDf, models)
                    filteredDistrDf = filterDf(val, distrDf, models)
                    break
                print("Model doesnt exist or was chosen already. Enter again")
    print("Models entered. Beginning visualization process")
    assert type(filteredScoreDf) is not list
    assert not filteredScoreDf.empty
    assert not filteredDistrDf.empty
    return filteredScoreDf, filteredDistrDf, filteredSplitDistrDf


def printDfStats(df):
    sumScore = {}
    avg = {}
    median = {}
    std = {}

    for filtered in df["id"].unique():
        score = np.sum(df[df["id"] == filtered][snapshotScoreKey])
        sumScore[filtered] = score
        avg[filtered] = np.average(df[snapshotScoreKey])
        median[filtered] = np.median(df[snapshotScoreKey])
        std[filtered] = np.std(df[snapshotScoreKey])
    sorted_data = dict(sorted(sumScore.items(), key=lambda item: item[1]))
    print("best scores", sorted_data)
    sorted_data2 = dict(sorted(avg.items(), key=lambda item: item[1]))
    print("\navg scores", sorted_data2)
    sorted_data3 = dict(sorted(median.items(), key=lambda item: item[1]))
    print("\nmedian scores", sorted_data3)
    sorted_data4 = dict(sorted(std.items(), key=lambda item: item[1]))
    print("\nstd scores", sorted_data4)


def plotMultipleLineplots(df, hue="id"):
    sns.color_palette("tab10")
    if args.scores == "both":
        yColumns = ["snapshotScore", "bestCurricScore"]
    elif args.scores == "curric":
        yColumns = ["bestCurricScore"]
    else:
        yColumns = ["snapshotScore"]
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.set_theme(style="darkgrid")
    # printDfStats(df) # TODO ? maybe args or just remove this part
    for col in yColumns:
        sns.lineplot(data=df, x='iterationSteps', y=col, hue=hue, ax=ax, errorbar=args.errorbar, palette="tab10")

    ax.set_ylabel("Average Reward", fontsize=labelFontsize)
    ax.set_xlabel("Iterations", fontsize=labelFontsize)

    # box = ax.get_position()
    # Edit this out to move the legend out of the plot
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    minX = 0
    maxX = args.xIterations + OFFSET
    assert minX < maxX, "min X must be smaller than max X"
    ax.set_xlim((0, args.xIterations + OFFSET))
    ax.set_ylim((0, 1))
    legendTitle = ""
    if hue != "id":
        legendTitle = hue
    plt.legend(loc="best", fontsize=labelFontsize, title=legendTitle)
    plt.title("Evaluation Performance in all Environments", fontsize=titleFontsize)
    plt.yticks(fontsize=tickFontsize)
    plt.xticks(fontsize=tickFontsize)
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


def removeSuffixCharInLegend(ax, columnsToVisualize):
    legend = ax.legend(fontsize=labelFontsize)
    labels = []
    sizes = [s[:-1] if s.endswith('n') else s for s in columnsToVisualize]
    for item in legend.get_texts():
        label = item.get_text()
        for size in sizes:
            if label.startswith(size):
                labels.append(size)
                break
    for label, text in zip(legend.texts, labels):
        label.set_text(text)


def showDistrVisualization(df, columnsToVisualize, title="Environment Distribution", isSplit=False):
    fig, ax = plt.subplots(figsize=(12, 8))
    if isSplit:
        # TODO only works with 1 experiment unfortunately (or aggregates it all)
        assert args.normalize

        def filter_entries(entry):
            return entry not in ['id', 'trained until'] and entry.endswith('n')

        cols = list(filter(filter_entries, df.columns))

        df = df.melt(id_vars=['trained until', 'id'],
                     value_vars=cols,
                     var_name='Environment',
                     value_name='Value')
        sns.barplot(x='trained until', y='Value', hue='Environment', data=df)
        plt.title("Environment Distribution at different Training Stages", fontsize=titleFontsize)
        plt.ylim((0, 1))
        removeSuffixCharInLegend(ax, columnsToVisualize)
        plt.show()
    else:
        group_col = "group" if 'group' in df.columns else 'DataFrame'
        # sort by the numerical values of the string (50k, 75k, ...)
        if args.rhea or args.nsga or args.ga:
            df['sort_col'] = df['id'].str.split('_', n=1, expand=True)[0].str.replace('k', '').astype('int')
        elif args.crossoverMutation:
            # add the numerical parts together (cross54_mut_56 => 54 + 56 = 110)
            df['sort_col'] = df['id'].str.extractall(r'(\d+)').astype(int).sum(level=0)
        else:
            # default: only sort by the ID string
            df["sort_col"] = df["id"]
        df = df.sort_values(by=['sort_col', group_col])
        df = df.drop('sort_col', axis=1)
        grouped_df = df[columnsToVisualize].groupby('id').agg(['mean', 'std'])
        grouped_df = grouped_df.reset_index()
        melted_df = grouped_df.melt(id_vars='id', var_name=['Column', 'Statistic'], value_name='Value')
        melted_df['id'] = pd.Categorical(melted_df['id'], categories=df['id'].unique(), ordered=True)
        # TODO vllt mit pivot table machen um das mit errorbar wieder hinzukriegen
        sns.barplot(data=melted_df[melted_df["Statistic"] == "mean"], x='id', y='Value', hue='Column', errorbar=args.errorbar)
        # TODO filter x ticks
        plt.ylabel('Occurence', fontsize=labelFontsize)
        plt.title(title, fontsize=titleFontsize)
        plt.xlabel('')
        plt.yticks(fontsize=tickFontsize)
        plt.xticks(rotation=-30, ha='left', fontsize=labelFontsize - 2)
        plt.subplots_adjust(bottom=0.2)
        if args.normalize:
            plt.ylim((0, 1))

        removeSuffixCharInLegend(ax, columnsToVisualize)
        plt.show()


def includeNormalizedColumns(aggregatedDf, prefix):
    """
    Updates the df to include the normalized columns of the environment distributions
    :param aggregatedDf: DataFrame to update
    :param prefix: prefix to use for column names
    :return: updated DataFrame
    """
    if not isDoorKey:
        envSizes = ['5x5', '6x6', '8x8', '16x16']
    else:
        envSizes = ['6x6', '8x8', '10x10', '12x12']
    normalizedDistributions = {size: [] for size in envSizes}
    # for splitdistr
    if type(prefix) == list:
        envSizes = prefix
        normalizedDistributions = {size: [] for size in envSizes}
        for i, row in aggregatedDf.iterrows():
            for j, env in enumerate(envSizes):
                normalizedDistributions[envSizes[j]].append(1.0 * row[f"{envSizes[j]}"])
        total = 0
        for env in normalizedDistributions:
            total += sum(normalizedDistributions[env])
        for env in normalizedDistributions:
            normalizedDistributions[env] = sum(normalizedDistributions[env]) / total
        for size in envSizes:
            column_name = f"{size}n"
            aggregatedDf[column_name] = normalizedDistributions[size]
    else:
        for i, row in aggregatedDf.iterrows():
            environmentTotalOccurence = sum(row[f"{size}{prefix}"] for size in envSizes)
            for size in envSizes:
                normalizedDistributions[size].append(1.0 * row[f"{size}{prefix}"] / environmentTotalOccurence)
        for size in envSizes:
            column_name = f"{size}n"
            aggregatedDf[column_name] = normalizedDistributions[size]
    return aggregatedDf


def showTrainingTimePlot(aggregatedDf):
    fig, ax = plt.subplots(figsize=(12, 8))
    group_col = "group" if 'group' in aggregatedDf.columns else 'DataFrame'  # TODO probably not for every experiment
    if args.rhea or args.nsga or args.ga:
        aggregatedDf['sort_col'] = aggregatedDf['id'].str.split('_', n=1, expand=True)[0].str.replace('k', '').astype('int')
        aggregatedDf = aggregatedDf.sort_values(by=['sort_col', group_col])
        aggregatedDf = aggregatedDf.drop('sort_col', axis=1)
        sns.barplot(x='id', y='sumTrainingTime', hue=group_col, dodge=False, data=aggregatedDf, ax=ax)
    else:
        sns.barplot(x='id', y='sumTrainingTime', data=aggregatedDf, ax=ax)

    plt.ylabel('training time (hours)', fontsize=labelFontsize)
    plt.xlabel('')
    title = "Training Time"
    # ax.legend(loc='upper right', bbox_to_anchor=(0.5, -0.2), fontsize="14") # TODO fix fontsize

    if args.title:
        title = args.title
    plt.title(title, fontsize=titleFontsize)
    plt.xticks(rotation=-45, ha='left', fontsize=tickFontsize)
    plt.subplots_adjust(bottom=0.3)

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


def handleDistributionVisualization(df):
    column_prefixes = {'snapshotDistr': 's', 'curricDistr': 'c', 'allDistr': 'a'}
    for arg, prefix in column_prefixes.items():
        if getattr(args, arg):
            if args.normalize:
                df = includeNormalizedColumns(df, prefix)
                if isDoorKey:
                    columns_to_visualize = [f'6x6n', f'8x8n', f'10x10n', f'12x12n', 'id']
                else:
                    columns_to_visualize = [f'5x5n', f'6x6n', f'8x8n', f'16x16n', 'id']
                title = "Normalized Environment Distributions"
            else:
                if isDoorKey:
                    columns_to_visualize = [f'6x6{prefix}', f'8x8{prefix}', f'10x10{prefix}', f'12x12{prefix}', 'id']
                else:
                    columns_to_visualize = [f'5x5{prefix}', f'6x6{prefix}', f'8x8{prefix}', f'16x16{prefix}', 'id']

                prefix = ""
                if args.curricDistr:
                    prefix = "Best Curricula "
                elif args.snapshotDistr:
                    prefix = "Best First Step "
                title = prefix + "Environment Distribution"
            showDistrVisualization(df, columns_to_visualize, title)


def plotAggregatedBarplot(df):
    assert type(df) is not list and not df.empty
    """ if not args.crossoverMutation:
        for f in df:
            for fullModelName in f["id"]:
                expParams = fullModelName.split("_")
                noModelName = "_".join(expParams[1:])
                break
            f["id"] = noModelName + "_" + expParams[0]
    """

    if args.trainingTime:
        showTrainingTimePlot(df)
    if args.splitDistr:
        if not isDoorKey:
            toVisualize = ["MiniGrid-Dynamic-Obstacles-5x5", "MiniGrid-Dynamic-Obstacles-6x6", "MiniGrid-Dynamic-Obstacles-8x8",
                           "MiniGrid-Dynamic-Obstacles-16x16"]
        else:
            toVisualize = ["MiniGrid-DoorKey-6x6", "MiniGrid-DoorKey-8x8", "MiniGrid-DoorKey-10x10",
                           "MiniGrid-DoorKey-12x12"]  # todo i probably have to cut this in result so i can remove the minigrid prefix from this
        if args.normalize:
            df = includeNormalizedColumns(df, toVisualize)
        showDistrVisualization(df, toVisualize, "Split Environment Distribution", True)
    else:
        handleDistributionVisualization(df)


def getGetNrFromModelName(modelName):
    splitModelName = modelName.split("_")
    for sub in splitModelName:
        if "gen" in sub:
            return int(sub[0])
    raise Exception(f"Could not get gen nr from modelname {modelName}")


def showFilteredGenPlot(df):
    genColumn = "nGen"
    df[genColumn] = df['id'].apply(getGetNrFromModelName)
    nGenDict = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
    usedIds = []
    for i, row in df.iterrows():
        id = row["id"]
        if id not in usedIds:
            usedIds.append(id)
            genNr = str(getGetNrFromModelName(id))
            nGenDict[genNr] += 1
    print("nGen Dict", nGenDict)
    df = df[df[genColumn] <= 3]
    df = df[df[genColumn] >= 1]
    plotMultipleLineplots(df, genColumn)


def showFilteredIterationSteps(df):
    iterationStepsDict = {"25000": 0, "50000": 0, "75000": 0, "100000": 0, "150000": 0, "250000": 0}
    usedIds = []
    for i, row in df.iterrows():
        id = row["id"]
        if id not in usedIds:
            usedIds.append(id)
            iterStep = row["group"]
            iterationStepsDict[str(iterStep)] += 1
    print("iterations Dict", iterationStepsDict)
    df = df[df["group"] != 25000]
    df = df[df["group"] != 250000]
    df["iterationSteps "] = df["group"]  # TODO idk why i originally named this group
    plotMultipleLineplots(df, "iterationSteps ")


def getCurricCountFromModelName(modelName):
    """

    :param modelName:
    :return:
    """
    splitModelName = modelName.split("_")
    for sub in splitModelName:
        if "curric" in sub:
            return int(sub[0])
    raise Exception(f"Could not get curric count from modelname {modelName}")


def showFilteredCurricCount(df):
    curricCountColumn = "curricCount"
    df[curricCountColumn] = df['id'].apply(getCurricCountFromModelName)
    curricCountDict = defaultdict(int)
    usedIds = []
    for i, row in df.iterrows():
        id = row["id"]
        if id not in usedIds:
            usedIds.append(id)
            curricCountDict[str(getCurricCountFromModelName(id))] += 1
    df = df[df[curricCountColumn] != 7]
    print("curricCount Dict", curricCountDict)
    plotMultipleLineplots(df, curricCountColumn)


def getCurricLenFromModelName(modelName):
    splitModelName = modelName.split("_")
    for sub in splitModelName:
        if "step" in sub:
            return int(sub[0])
    raise Exception(f"Could not get curric step from modelname {modelName}")


def showFilteredCurricLen(df):
    curricLenCol = "curricLen"
    df[curricLenCol] = df['id'].apply(getCurricLenFromModelName)
    curricLenDict = defaultdict(int)
    usedIds = []
    for i, row in df.iterrows():
        id = row["id"]
        if id not in usedIds:
            usedIds.append(id)
            genNr = str(getCurricLenFromModelName(id))
            curricLenDict[genNr] += 1
    df = df[df[curricLenCol] != 7]
    print("curricLen Dict", curricLenDict)
    plotMultipleLineplots(df, curricLenCol)


def showGroupedScorePlots(filteredScoreDf):
    if args.gen:
        showFilteredGenPlot(filteredScoreDf)
    if args.iterGroup:
        showFilteredIterationSteps(filteredScoreDf)
    if args.curricCount:
        showFilteredCurricCount(filteredScoreDf)
    if args.curricLen:
        showFilteredCurricLen(filteredScoreDf)


def showDistributionPlots(filteredSplitDistrDf, filteredFullDistrDf):
    if args.splitDistr:
        plotAggregatedBarplot(filteredSplitDistrDf)
    else:
        plotAggregatedBarplot(filteredFullDistrDf)


def main(comparisons: int):
    sns.set(font_scale=2)
    pathList = ["storage", "_evaluate"]
    evalDirBasePath = os.path.join(*pathList)
    statusJson = "status.json"

    expeirmentsWithLogFilePaths = defaultdict(list)
    for dirpath, dirnames, filenames in os.walk(evalDirBasePath):
        if statusJson in filenames:
            helper = dirpath.split("\\")[-2]
            pathToJson = os.path.join(dirpath, statusJson)
            # Append pathToJson to the list associated with helper
            expeirmentsWithLogFilePaths[helper].append(pathToJson)

    scoreDf, distrDf, splitDistrDf = getAllDfs(expeirmentsWithLogFilePaths)
    models = scoreDf["id"].unique()
    sns.set_theme(style="darkgrid")

    filteredScoreDf, filteredFullDistrDf, filteredSplitDistrDf = \
        getUserInputForMultipleComparisons(models, comparisons, scoreDf, distrDf, splitDistrDf)

    if args.scores is not None:
        plotMultipleLineplots(filteredScoreDf)

    showGroupedScorePlots(filteredScoreDf)
    showDistributionPlots(filteredSplitDistrDf, filteredFullDistrDf)

    # TODO DF save as csv
    # filepath = Path('./out.csv')
    # filepath.parent.mkdir(parents=True, exist_ok=True)
    # scoreDf.to_csv(filepath, index=False)
    print("Evaluaiton finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="DoorKey", help="Whether to use doorkey or dynamic obstacle or both")
    parser.add_argument("--title", default=None, type=str, help="Title of the distribution plots")
    parser.add_argument("--comparisons", default=-1, help="Choose how many models you want to compare")
    parser.add_argument("--crossoverMutation", action="store_true", default=False, help="Select the crossovermutation varied experiments")
    parser.add_argument("--splitDistr", action="store_true", default=False, help="Whether to show the split distribution (in 1kk steps)")

    parser.add_argument("--trainingTime", action="store_true", default=False, help="Show training time plots")
    parser.add_argument("--snapshotDistr", action="store_true", default=False, help="Show first step distribution plots")
    parser.add_argument("--curricDistr", action="store_true", default=False, help="show all best curricula distributions plots")
    parser.add_argument("--allDistr", action="store_true", default=False, help="Show all distribution plots")
    parser.add_argument("--scores", default=None, help="Whether to plot snapshot, curric or both scores")
    parser.add_argument("--normalize", action="store_true", default=False, help="Whether or not to normlaize the env distributions")

    parser.add_argument("--iter", default=0, type=int, help="filter for iterations")
    parser.add_argument("--xIterations", default=1100000, type=int, help="#of iterations to show on the xaxis")
    parser.add_argument("--steps", action="store_true", default=False, help="filter for #curricSteps")
    parser.add_argument("--gen", action="store_true", default=False, help="Whether to filter #gen")
    parser.add_argument("--iterGroup", action="store_true", default=False, help="Whether to group by iterationSteps")
    parser.add_argument("--curricLen", action="store_true", default=False, help="Whether to group by curricLength")
    parser.add_argument("--curricCount", action="store_true", default=False, help="Whether to group by curricCount")

    parser.add_argument("--rhea", action="store_true", default=False, help="Only using rhea runs")
    parser.add_argument("--nsga", action="store_true", default=False, help="Only using GA runs")
    parser.add_argument("--ga", action="store_true", default=False, help="Only using NSGA runs")
    parser.add_argument("--rrh", action="store_true", default=False, help="Include RRH runs, even if --rhea was speicifed")
    parser.add_argument("--rrhOnly", action="store_true", default=False, help="Only RRH runs")
    parser.add_argument("--showCanceled", action="store_true", default=False, help="Whether to use canceled runs too")
    parser.add_argument("--errorbar", default=None, type=str, help="What type of errorbar to show on the lineplots. (Such as sd, ci etc)")
    args = parser.parse_args()
    args.rhea = args.curricLen or args.iterGroup or args.gen or args.curricCount
    args.filter = args.comparisons == -1 and (
            args.crossoverMutation or args.splitDistr or
            args.iter or args.steps or args.rrhOnly or args.rhea or args.nsga or args.ga)
    isDoorKey = args.env is None or "door" in args.env
    main(int(args.comparisons))
