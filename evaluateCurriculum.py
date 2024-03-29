import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import natsort

from curricula.Result import Result
from utils.curriculumHelper import *

OFFSET = 1000


def getSpecificModel(specificModelList: list, modelName: str):
    """
    Returns the performance and misc. info dfs as tuple for the given modellist and modelname
    """
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
        if args.crossoverMutation and "_c" and "Mut" not in modelName and "Cross" not in modelName:
            continue
        if not isDoorKey and "Doorkey" in jsonPaths[0]:
            continue
        elif isDoorKey and "DynamicObstacle" in jsonPaths[0]:
            continue
        if args.rrhOnly and "RndRH" not in jsonPaths[0]:
            continue
        # RRH Filter ?
        # TODO further filters ???
        tmpScoreDf, tmpDistrDf, tmpSplitDf = getSpecificModel(jsonPaths, modelName)
        scoreDf = pd.concat([scoreDf, tmpScoreDf], ignore_index=True)
        fullDistrDf = pd.concat([fullDistrDf, tmpDistrDf], ignore_index=True)
        splitDistrDf = pd.concat([splitDistrDf, tmpSplitDf], ignore_index=True)
    scoreDf = scoreDf[scoreDf[iterationSteps] < args.xIterations + OFFSET]
    assert not scoreDf.empty
    assert not fullDistrDf.empty
    return scoreDf, fullDistrDf, splitDistrDf


def filterDf(filters: list[str], df, models, showCanceled=False):  # TODO remove param
    """
    Given a list of models and the main dataframe, it filters all the relevant id columns matching the @val prefix
    """
    if df.empty:
        print("---Empty df!---")
        return df

    if args.all:
        return df

    def passes_filters(colId, filterList):
        """
        Helper method to filtere the specified items out of the df
        :param colId:
        :param filterList:
        :return:
        """
        for filterOption in filterList:
            if filterOption in colId:
                if (filterOption == 'GA' and 'NSGA' in colId) or \
                        (filterOption == '50k' and ('150k' in colId or '250k' in colId)) or \
                        ("GA" in filterOption and "const" in colId) or \
                        ("250k" in colId) or \
                        (not showCanceled and "C_" in colId):  # TODO maybe args for const param
                    return False
            else:
                return False
        return True
    filteredDf = df[df['id'].apply(lambda x: passes_filters(x, filters))]
    return filteredDf


def getStep(tmp):
    """
    Returns the curriculum len from the given string. 3step -> 3
    """
    for s in tmp:
        if "step" in s:
            return s[0]
    raise Exception("could not extract step")


def getCurric(tmp):
    """
    Returns the curric amount from the given string. 3curric -> 3
    """
    for s in tmp:
        if "curric" in s:
            return s[0]
    raise Exception("could not extract curric")


def getGen(tmp):
    """
    Returns the gen from the given string ("3gen") -> 3
    """
    for s in tmp:
        if "gen" in s:
            return s[0]
    raise Exception("could not extract gen")


def getIterstep(tmp):
    """
    Returns the itertion step from the given string. 100k -> 100
    """
    for s in tmp:
        if "k" in s:
            return s[:-1]
    raise Exception("could not extract iterstep")


def getUserInputForMultipleComparisons(models: list, comparisons: int, scoreDf, distrDf, splitDistrDf) -> tuple:
    """
    Method to get the current selection of models that will be used for the analysis. Either be user input or
    by pre-specified filters
    :param models: the list of the model names
    :param comparisons: how many models the user can select
    :param scoreDf: the performance DF
    :param distrDf: the df containing the env distributions and training time
    :param splitDistrDf: the df containing the split env distribution, 1m for each step
    :return:
    """
    modelsEntered: int = 0
    usedModels = []
    filteredScoreDf = pd.DataFrame()
    filteredDistrDf = pd.DataFrame()
    filteredSplitDistrDf = pd.DataFrame()
    if args.filter:
        filters = []
        if args.rhea:
            # slightly hacky to avoid the "GA" == duplicate check above (since NSGA is also in GA string)
            filters.append("GA_")
            if args.rrh:
                filters.append("RRH")
        if args.ga:
            filters.append("GA")
        if args.nsga:
            filters.append("NSGA")
        if args.rrhOnly:
            filters.append("RndRH")
        if args.ppoOnly:
            filters.append("PPO")
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
                filteredScoreDf = pd.concat([filteredScoreDf, scoreDf[scoreDf["id"] == models[int(val)]]],
                                            ignore_index=True)
                filteredDistrDf = pd.concat([filteredDistrDf, distrDf[distrDf["id"] == models[int(val)]]],
                                            ignore_index=True)
                filteredSplitDistrDf = pd.concat(
                    [filteredSplitDistrDf, splitDistrDf[splitDistrDf["id"] == models[int(val)]]], ignore_index=True)
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
    """
    Helper method to optionally filter the best runs of the given df
    Also prints the median an average scores of the experiments in the df
    :return:
    """
    sumScore = {}
    avg = {}
    median = {}
    std = {}

    for filtered in df["id"].unique():
        score = df[df["id"] == filtered][snapshotScoreKey]
        sumScore[filtered] = np.sum(score)
        avg[filtered] = round(np.average(score), 3)
        median[filtered] = round(np.median(score), 3)
        std[filtered] = round(np.std(score), 3)
    averageScore = dict(sorted(avg.items(), key=lambda item: item[1]))
    print("\navg scores", averageScore)
    medianScore = dict(sorted(median.items(), key=lambda item: item[1]))
    print("\nmedian scores", medianScore)

    tmpDf = pd.DataFrame(avg.items(), columns=['id', 'average'])
    filtered_df = df
    # filtered_df = tmpDf[tmpDf['average'] >= 0.84]  # TODO dont use this if there is few runs
    ids = (filtered_df["id"].unique())
    res = {}
    j = 0
    for i in averageScore.items():
        res[i[0]] = i[1]
        j += 1
    j = 0
    for i in medianScore.items():
        res[i[0]] += i[1]
        j += 1
    sortedS = dict(sorted(res.items(), key=lambda item: item[1]))
    for r in sortedS.items():
        print(r)
    print("unique", len(ids))
    return ids


def transformRSLabel(label):
    tmp = label[3:]
    if "nRS" in tmp:
        tmp = "No Reward Shaping"
    else:
        tmp = "Reward Shaping"
    return tmp


def transformGammaLabel(label):
    # TODO ?
    return label
    tmp = label.split("_")[-1]
    assert "gamma" in tmp
    transformedLabel = "Gamma 0." + tmp[-2]
    return transformedLabel


def transformNsgaVsGa(label):
    if "NSGA" in label:
        return "NSGA-II"
    elif "GA" in label:
        return "GA"
    raise Exception("Incorrect label for transformation")


def transformEnvLabel(label, groupRHEA=False):
    if "AP" in label or "AllPara" in label:
        return "All Parallel"
    if "SPCL" in label:
        return "SPCL"
    if "RndRH" in label:
        return "Random RH"  # comment this line if you dont want to group by RRH
        splitStr = label.split("_")[2:]
        resultStr = ""
        for split in splitStr:
            resultStr += split
            if "curric" in split:
                break
            resultStr += " "
        return resultStr
    if "PPO" in label:
        tmp = label.split("_")
        return tmp[0] + " " + tmp[1]
    # cut the initials, like '1_GA_"
    if groupRHEA:
        return "RHEA CL"
    result = label[5:]
    result = result.split("_")
    transformedLabel = ""
    for r in result:
        transformedLabel += r
        if "k" in r:
            return r
        if "curric" in r:
            break
        transformedLabel += " "
        # Add additional space to make 50k and 150k appear same length in legend
        if r == result[0] and len(r) == 3:
            transformedLabel += "  "
    transformedLabel = ""
    tmp = label.split("_")
    for part in tmp:
        if "k" in part or "step" in part or "gen" in part:
            transformedLabel += part + " "
    if tmp[0] == "C":
        transformedLabel += "(C)"
    assert transformedLabel != "", f"{label}"
    return transformedLabel


def transformMultiObjLabel(label):
    if "MultiObj_nRS" in label:
        return "Multi Objective Variant"
    elif "_RS" in label:
        return "Single Objective Variant"
    raise Exception(f"transformation failed for {label}")


def transformCrossoverMutationLabel(label):
    tmp = label.split("_")
    result = "Mutation: 0." + tmp[0][-2:] + " " + "Crossover: 0." + tmp[1][-2:]
    return result


def plotMultipleLineplots(df, hue="id", legendLoc="lower right"):
    ids = printDfStats(df)
    df = df[df["id"].isin(ids)]
    assert len(df["id"].unique()) != 0, f"df: {df}"
    if args.title is not None:
        if "multi" in args.title:
            df['id'] = df['id'].map(transformMultiObjLabel)
        elif "Gamma" in args.title:
            df['id'] = df['id'].map(transformGammaLabel)
        elif "Reward Shaping" in args.title:
            df['id'] = df['id'].map(transformRSLabel)
        elif "Dyn" in args.title or "Door" in args.title:
            df['id'] = df['id'].map(transformEnvLabel)
        elif "Performance of" in args.title:
            df["id"] = df["id"].map(transformEnvLabel)
        elif "Crossover" in args.title:
            df["id"] = df["id"].map(transformCrossoverMutationLabel)
    if args.scores == "both":
        yColumns = ["snapshotScore", "bestCurricScore"]
    elif args.scores == "curric":
        yColumns = ["bestCurricScore"]
    else:
        yColumns = ["snapshotScore"]
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.set_theme(style="darkgrid")
    for col in yColumns:
        # todo for the gamma runs !!!
        """
        # df["style"] = df[df["id"].str.contains("snapshotScore")]
        df["tmp"] = df["id"].astype("str") + "_" + col
        if "snapshotScore" in col:
            palette = sns.color_palette("Set1", n_colors=1 + int(args.comparisons))

        else:
            palette = sns.color_palette("tab10", n_colors=5)
        """
        # df1 = df[df["id"].str.contains("C_")]
        # df2 = df[~df["id"].str.contains("C_")]
        ax = sns.lineplot(data=df, x='iterationSteps', y=col, hue=hue, ax=ax, errorbar=args.errorbar, palette=palette,
                          linestyle="-"
                          )
        # ax = sns.lineplot(data=df2, x='iterationSteps', y=col, hue=hue, ax=ax, errorbar=args.errorbar, palette=palette,
        #                 linestyle="-"
        #                )
        # ax.lines[0].set_linestyle("--")
        """lines = ax.lines

        for line in lines:
            if "child" in line.get_label():
                continue
            print(line.get_label())
            if "3step" in line.get_label():
                line.set_linestyle("-")  # Solid line
            else:
                line.set_linestyle("--")  # Dashed line"""

    handles, labels = ax.get_legend_handles_labels()
    sorted_labels = natsort.natsorted(labels)
    sorted_handles = [handles[labels.index(label)] for label in sorted_labels]
    ax.set_ylabel("Average Reward", fontsize=labelFontsize)
    ax.set_xlabel("Iterations", fontsize=labelFontsize)

    # box = ax.get_position()
    # Edit this out to move the legend out of the plot
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    if isDoorKey:
        minY = 0
    else:
        minY = -1
    maxX = args.xIterations + OFFSET
    minX = 0
    assert minX < maxX, "min X must be smaller than max X"
    ax.set_xlim((minX, args.xIterations + OFFSET))
    ax.set_ylim((minY, 1))

    legendTitle = ""
    if hue != "id":
        legendTitle = hue

    ax.legend(sorted_handles, sorted_labels, loc=legendLoc, fontsize=labelFontsize, title=legendTitle)

    title = args.title or "Evaluation Performance in all Environments"
    plt.title(title, fontsize=titleFontsize)
    plt.tight_layout()
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
    else:
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
    sizes = [s[:-1] if (s.endswith('n') or s.endswith('c')) else s for s in columnsToVisualize]
    for item in legend.get_texts():
        label = item.get_text()
        for size in sizes:
            if label.startswith(size):
                labels.append(size)
                break
    for label, text in zip(legend.texts, labels):
        label.set_text(text)


def plotNSGAMultivsSingleDistribution(df, ax, columnsToVisualize, title):
    def transform_label(label):
        if "MultiObj_nRS" in label or "multi" in label:
            return "Multi Objective Variant"
        elif "_RS" in label:
            return "Single Objective Variant"
        raise Exception(f"transformation failed for {label}")

    df['id'] = df['id'].map(transform_label)

    subset_df = df[columnsToVisualize]
    subset_df = subset_df.melt(id_vars='id', var_name='Column', value_name='Value')
    # Remove the 'n' at the end of the column names
    subset_df['Column'] = subset_df['Column'].str.rstrip('n')
    sns.barplot(x='Column', y='Value', hue='id', data=subset_df, errorbar="se")
    plt.xlabel('Environment', fontsize=labelFontsize)
    plt.ylabel('Occurence', fontsize=labelFontsize)
    if args.normalize:
        plt.ylim((0, 1))
    plt.title(title, fontsize=titleFontsize)
    plt.yticks(fontsize=tickFontsize)
    plt.xticks(fontsize=labelFontsize)
    removeSuffixCharInLegend(ax, columnsToVisualize)

    plt.show()
    """
    # the switched x axis and legend variant
            subset_df = df[['id', '6x6n', '8x8n', '10x10n', '12x12n']]
    subset_df = subset_df.melt(id_vars='id', var_name='Column', value_name='Value')

    sns.barplot(x='Column', y='Value', hue='id', data=subset_df, errorbar="se")

    plt.xlabel('Column Names')
    plt.ylabel('Values')
    plt.title('Bar Plot')
    plt.show()
    """


def showSplitEnvDistribution(df, ax, columnsToVisualize):
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
    plt.tight_layout()
    plt.show()


def showDistrVisualization(df, columnsToVisualize, title="Environment Distribution", isSplit=False):
    fig, ax = plt.subplots(figsize=(12, 8))

    df['id'] = df['id'].map(transformEnvLabel)
    # df["id"] = df["id"].map(transformNsgaVsGa)

    if isSplit:
        showSplitEnvDistribution(df, ax, columnsToVisualize)
    else:
        group_col = "group" if 'group' in df.columns else 'DataFrame'  # TODO ?
        # sort by the numerical values of the string (50k, 75k, ...)
        if args.rhea or args.nsga or args.ga:
            df['sort_col'] = df['id'].str.split('_', n=1, expand=True)[0].str.replace('k', '').astype('int')
        elif args.crossoverMutation:
            # add the numerical parts together (cross54_mut_56 => 54 + 56 = 110)
            df['sort_col'] = df['id'].str.extractall(r'(\d+)').astype(int).sum(level=0)
        else:
            # default: only sort by the ID string
            df["sort_col"] = df["id"]

    if "NSGA-II" in title and "Multi" in title:
        plotNSGAMultivsSingleDistribution(df, ax, columnsToVisualize, title)
    else:
        grouped_df = df[columnsToVisualize].groupby('id').agg(['mean', 'std'])
        grouped_df = grouped_df.reset_index()
        melted_df = grouped_df.melt(id_vars='id', var_name=['Column', 'Statistic'], value_name='Value')
        melted_df['id'] = pd.Categorical(melted_df['id'], categories=df['id'].unique(), ordered=True)
        sns.barplot(data=melted_df[melted_df["Statistic"] == "mean"], x='id', y='Value', hue='Column',
                    errorbar=args.errorbar)
        # TODO filter x ticks
        plt.ylabel('Occurence', fontsize=labelFontsize)
        plt.title(title, fontsize=titleFontsize)
        plt.xlabel('')
        plt.yticks(fontsize=tickFontsize)
        # plt.xticks(rotation=-15, ha='left', fontsize=labelFontsize - 2)
        plt.xticks(fontsize=labelFontsize - 2)
        plt.subplots_adjust(bottom=0.2)
        if args.normalize:
            plt.ylim((0, 1))

        removeSuffixCharInLegend(ax, columnsToVisualize)
        plt.tight_layout()
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
    """
    ['0_GA_75k_2step_3gen_3curric_nRS' '0_GA_75k_3step_3gen_3curric_RS'
 '2_GA_75k_3step_1gen_3curric_Door_RS'
 '2_GA_75k_3step_2gen_3curric_Door_RS_FF'
 '4_GA_75k_3step_1gen_3curric_nRS' '4_GA_75k_3step_2gen_3curric_nRS'
 'NSGA_75k_3step_2gen_4curric_RS'
 'NSGA_75k_3step_3gen_3curric_MultiObj_nRS'
 'NSGA_75k_3step_3gen_3curric_RS']

    """
    # aggregatedDf["id"] = aggregatedDf[aggregatedDf["id"].isin(getValidRuns(aggregatedDf["id"]))]
    aggregatedDf['id'] = aggregatedDf['id'].map(lambda x: transformEnvLabel(x, True))  # Example for boolean transformation
    aggregatedDf = aggregatedDf[~aggregatedDf["id"].str.contains("PPO 10x10")]

    print(aggregatedDf)

    fig, ax = plt.subplots(figsize=(12, 8))
    group_col = "group" if 'group' in aggregatedDf.columns else 'DataFrame'  # TODO probably not for every experiment
    # sns.barplot(x='id', y='sumTrainingTime', data=aggregatedDf, ax=ax)

    if args.rhea or args.nsga or args.ga:
        aggregatedDf['sort_col'] = aggregatedDf['id'].str.split('_', n=1, expand=True)[0].str.replace('k', '').astype(
            'int')
        aggregatedDf = aggregatedDf.sort_values(by=['sort_col', group_col])
        aggregatedDf = aggregatedDf.drop('sort_col', axis=1)
        sns.barplot(x='id', y='sumTrainingTime', hue=group_col, dodge=False, data=aggregatedDf, ax=ax)
    else:
        df = aggregatedDf.sort_values(by='sumTrainingTime', ascending=False)
        sns.barplot(x='id', y='sumTrainingTime', data=df, ax=ax)

    plt.ylabel('Training Time (Hours)', fontsize=labelFontsize)
    # ax.legend(loc='upper right', bbox_to_anchor=(0.5, -0.2), fontsize="14") # TODO fix fontsize

    if args.title:
        title = args.title
    else:
        title = " Training Time"

    plt.title(title, fontsize=titleFontsize)
    # plt.xticks(rotation=-45, ha='left', fontsize=tickFontsize)
    plt.xticks(fontsize=tickFontsize)
    # plt.subplots_adjust(bottom=0.3)

    # TODO the legend part might be specific to some of the settings and not universal

    """
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
    """
    ax.set_ylim((0, 96))
    plt.tight_layout()
    plt.legend([], [], frameon=False)
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
                if args.title:
                    title = args.title
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
                if args.title:
                    title = args.title
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
            toVisualize = ["MiniGrid-Dynamic-Obstacles-5x5", "MiniGrid-Dynamic-Obstacles-6x6",
                           "MiniGrid-Dynamic-Obstacles-8x8",
                           "MiniGrid-Dynamic-Obstacles-16x16"]
        else:
            toVisualize = ["MiniGrid-DoorKey-6x6", "MiniGrid-DoorKey-8x8", "MiniGrid-DoorKey-10x10",
                           "MiniGrid-DoorKey-12x12"]
        if args.normalize:
            df = includeNormalizedColumns(df, toVisualize)
        showDistrVisualization(df, toVisualize, "Split Environment Distribution", True)
    else:
        handleDistributionVisualization(df)


def getGenNrFromModelName(modelName):
    splitModelName = modelName.split("_")
    for sub in splitModelName:
        if "gen" in sub:
            return int(sub[0])
    raise Exception(f"Could not get gen nr from modelname {modelName}")


def showFilteredGenPlot(df):
    genColumn = "nGen"
    df[genColumn] = df['id'].apply(getGenNrFromModelName)
    nGenDict = defaultdict(int)
    usedIds = []
    for i, row in df.iterrows():
        id = row["id"]
        if id not in usedIds:
            usedIds.append(id)
            genNr = str(getGenNrFromModelName(id))
            nGenDict[genNr] += 1
    print("nGen Dict", nGenDict)
    t = 4 == 4
    if t:
        # df = df[df[genColumn] < 6]
        plotMultipleLineplots(df, genColumn)
    else:
        # df = df[df["iterationSteps"] <= 333333]
        df = df[df[genColumn] == 5]
        # import time
        # from pathlib import Path
        # filepath = Path(f'./out_{time.time()}.csv')
        # filepath.parent.mkdir(parents=True, exist_ok=True)
        # df.to_csv(filepath, index=False)
        plotMultipleLineplots(df, )
        plotMultipleLineplots(df, genColumn)


def showFilteredIterationSteps(df):
    iterationStepsDict = defaultdict(int)
    usedIds = []
    for i, row in df.iterrows():
        id = row["id"]
        if id not in usedIds:
            usedIds.append(id)
            iterStep = row["group"]
            iterationStepsDict[str(iterStep)] += 1
    print("iterations Dict", iterationStepsDict)
    groupName = "iterationSteps "
    df[groupName] = df["group"]
    plotMultipleLineplots(df, groupName)


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
            curricCountDict[str(getCurricCountFromModelName(id)) + " curricula"] += 1
    ids = printDfStats(df)
    print("curricCount Dict", curricCountDict)
    t = 7 == 7
    if t:
        df = df[(df[curricCountColumn] != 3) | (df['id'].isin(ids))]
        plotMultipleLineplots(df, curricCountColumn)
    else:
        df = df[df[curricCountColumn] == 1]
        plotMultipleLineplots(df, )
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
    df = df[df[curricLenCol] != 7]  # only 1 run
    print("curricLen Dict", curricLenDict)
    t = 6 == 6
    if t:
        plotMultipleLineplots(df, curricLenCol)
    else:
        ids = printDfStats(df)
        df = df[(df['id'].isin(ids))]
        # df = df[(df[curricLenCol] != 3) | (df['id'].isin(ids))]
        df = df[df[curricLenCol] == 3]
        plotMultipleLineplots(df, )
        plotMultipleLineplots(df, curricLenCol)


def showGroupedScorePlots(filteredScoreDf):
    if args.gen:
        showFilteredGenPlot(filteredScoreDf)
    elif args.iterGroup:
        showFilteredIterationSteps(filteredScoreDf)
    elif args.curricCount:
        showFilteredCurricCount(filteredScoreDf)
    elif args.curricLen:
        showFilteredCurricLen(filteredScoreDf)
    elif args.scores is not None:
        plotMultipleLineplots(filteredScoreDf)


def showDistributionPlots(filteredSplitDistrDf, filteredFullDistrDf):
    if args.splitDistr:
        plotAggregatedBarplot(filteredSplitDistrDf)
    else:
        plotAggregatedBarplot(filteredFullDistrDf)


def main(comparisons: int):
    # sns.set(font_scale=2)
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

    showGroupedScorePlots(filteredScoreDf)
    showDistributionPlots(filteredSplitDistrDf, filteredFullDistrDf)
    print("Evaluaiton finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="doorKey", help="Whether to use doorkey or dynamic obstacle or both")
    parser.add_argument("--title", default=None, type=str, help="Title of the distribution plots")
    parser.add_argument("--comparisons", default=-1, help="Choose how many models you want to compare")
    parser.add_argument("--crossoverMutation", action="store_true", default=False,
                        help="Select the crossovermutation varied experiments")
    parser.add_argument("--splitDistr", action="store_true", default=False,
                        help="Whether to show the split distribution (in 1kk steps)")

    parser.add_argument("--trainingTime", action="store_true", default=False, help="Show training time plots")
    parser.add_argument("--snapshotDistr", action="store_true", default=False,
                        help="Show first step distribution plots")
    parser.add_argument("--curricDistr", action="store_true", default=False,
                        help="show all best curricula distributions plots")
    parser.add_argument("--allDistr", action="store_true", default=False, help="Show all distribution plots")
    parser.add_argument("--scores", default=None, help="Whether to plot snapshot, curric or both scores")
    parser.add_argument("--normalize", action="store_true", default=False,
                        help="Whether or not to normlaize the env distributions")

    parser.add_argument("--iter", default=0, type=int, help="filter for iterations")
    parser.add_argument("--xIterations", default=5000000, type=int, help="#of iterations to show on the xaxis")
    parser.add_argument("--steps", action="store_true", default=False, help="filter for #curricSteps")
    parser.add_argument("--gen", action="store_true", default=False, help="Whether to filter #gen")
    parser.add_argument("--iterGroup", action="store_true", default=False, help="Whether to group by iterationSteps")
    parser.add_argument("--curricLen", action="store_true", default=False, help="Whether to group by curricLength")
    parser.add_argument("--curricCount", action="store_true", default=False, help="Whether to group by curricCount")

    parser.add_argument("--rhea", action="store_true", default=False, help="Only using rhea runs")
    parser.add_argument("--all", action="store_true", default=False, help="Skip filtering step")
    parser.add_argument("--nsga", action="store_true", default=False, help="Only using GA runs")
    parser.add_argument("--ga", action="store_true", default=False, help="Only using NSGA runs")
    parser.add_argument("--rrh", action="store_true", default=False,
                        help="Include RRH runs, even if --rhea was speicifed")  # todo remove this part
    parser.add_argument("--rrhOnly", action="store_true", default=False, help="Only RRH runs")  # todo doesnt work anymore ?
    parser.add_argument("--showCanceled", action="store_true", default=False, help="Whether to use canceled runs too")
    parser.add_argument("--ppoOnly", action="store_true", default=False, help="Whether to show only ppo runs")
    parser.add_argument("--errorbar", default=None, type=str,
                        help="What type of errorbar to show on the lineplots. (Such as sd, ci etc)")
    # TODO remove old filter options
    args = parser.parse_args()
    args.rhea = args.rhea or args.curricLen or args.iterGroup or args.gen or args.curricCount
    args.filter = args.comparisons == -1 and (
            args.crossoverMutation or args.splitDistr or args.ppoOnly or args.all or
            args.iter or args.steps or args.rrhOnly or args.rhea or args.nsga or args.ga)
    isDoorKey = "door" in args.env
    # palette = sns.color_palette("tab10", n_colors=5)

    if args.comparisons == -1:
        args.comparisons = 20
    palette = sns.color_palette("Set1", n_colors=8)
    # palette = sns.color_palette("deep", n_colors=8)

    if args.ppoOnly:
        args.xIterations = 10000000
    main(int(args.comparisons))
