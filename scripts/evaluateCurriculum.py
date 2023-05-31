import os

import matplotlib.pyplot as plt

from scripts.Result import Result
from utils.curriculumHelper import *


def plotPerformance(allYValues: list[list[float]], allXValues: list[list[int]], maxReward: int, iterationsPerEnv: int, title: str,
                    modelNames: list[str]):
    minX = 0  # min([min(x) for x in xLists])
    maxX = max([max(x) for x in allXValues]) * 1.1

    # Create a figure and an axis
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
    """
    ax2.set_xlim(ax1.get_xlim())
    new_tick_locations = [(i * 2) * iterationsPerEnv for i in range(1, len(aList) // 2)]
    ax2.set_xticks(new_tick_locations)
    ax2.set_xticklabels([str(tick // 1000) + 'k' for tick in new_tick_locations]))
    """
    plt.show()


def plotSnapshotPerformance(results: list[Result], title: str):
    y = []
    for res in results:
        y.append(res.snapShotScores)

    x = []
    for res in results:
        x.append([])
        for i in range(res.epochsTrained):
            x[-1].append(i * res.iterationsPerEnv)
    maxReward = 1
    iterationsPerEnv = results[0].iterationsPerEnv
    modelNames = []
    for r in results:
        if r.iterationsPerEnv > iterationsPerEnv:
            iterationsPerEnv = r.iterationsPerEnv
        modelNames.append(r.modelName)
    print("highest iter env", iterationsPerEnv)

    plotPerformance(y, x, maxReward, iterationsPerEnv, title, modelNames)


def plotEpochAvgCurricReward(results: list[Result], title: str):
    y = []
    for res in results:
        y.append(res.avgEpochRewards)
    print("y:", y)
    maxReward = results[0].maxCurricAvgReward
    iterationsPerEnv = results[0].iterationsPerEnv
    plotPerformance(y, maxReward, iterationsPerEnv, title)


def plotBestCurriculumResults(results: list[Result], title: str):
    y = []
    for res in results:
        y.append(res.bestCurricScore)
    maxReward = results[0].curricMaxReward
    iterationsPerEnv = results[0].iterationsPerEnv
    for r in results:
        if r.iterationsPerEnv > iterationsPerEnv:
            iterationsPerEnv = r.iterationsPerEnv
    print("highest iter env", iterationsPerEnv)
    plotPerformance(y, maxReward, iterationsPerEnv, title)


def plotEnvsUsedDistribution(resultClassesList: list[Result], distributionType: str, titleInfo: str):
    if distributionType == SNAPTSHOTS_DISTR:
        envLists = resultClassesList[0].snapshotEnvDistribution
        pass
    elif distributionType == FULL_CURRIC_DISTR:
        envLists = resultClassesList[0].allCurricDistribution
        pass
    elif distributionType == BEST_CURRIC_DISTR:
        envLists = resultClassesList[0].bestCurriculaEnvDistribution
        pass
    else:
        raise Exception("Distribution type not found!")

    envDistribution = envLists
    # extract the numeric part of the keys of the envDistribution
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
    fig, ax = plt.subplots()
    bar_container = ax.bar(envs, envOccurrences)
    ax.set_ylabel('Occurrence')
    ax.set_title(titleInfo)
    ax.set_ylim(0, max(envOccurrences) * 1.1)

    # Add labels to the bars
    ax.bar_label(bar_container, labels=envOccurrences, fontsize=12, padding=5)
    plt.show()
    # TODO add colors again
    # TODO plot the snapshot vs curricReward problem
    # TODO find out a way to properly plot the difficulty list / maybe how it influences the results; and maybe how you can improve it so that it is not even needed in the first place
    # TODO find way to plot multiple models at once (and show some relevant legend for info of model name or sth like that)
    # TODO save the plots
    # TODO extra args option to only load 1 model
    # TODO experiment comparison: long iterPerStep with many Gen, with low iterPerStep with low gen


if __name__ == "__main__":
    evalDirectory = os.getcwd() + "\\storage\\save\\evaluate\\"
    logFilePaths = []
    evalDirectories = next(os.walk(evalDirectory))[1]
    for model in evalDirectories:
        logFilePaths.append(evalDirectory + model + "\\status.json")

    resultClasses = []
    for logFilePath in logFilePaths:
        if os.path.exists(logFilePath):
            with open(logFilePath, 'r') as f:
                trainingInfoDictionary = json.loads(f.read())
            assert trainingInfoDictionary is not None
            resultClass = Result(trainingInfoDictionary)
            resultClasses.append(resultClass)
        else:
            raise Exception(f"Path '{logFilePath}' doesnt exist!")

    # plotEnvsUsedDistribution(resultClasses, BEST_CURRIC_DISTR, "Distribution of envs of best performing curricula")
    # plotEnvsUsedDistribution(resultClasses, FULL_CURRIC_DISTR, "all Curric Distribution")
    # plotEnvsUsedDistribution(resultClasses, SNAPTSHOTS_DISTR, "snapshot Distribution")
    # TODO are there other distributions that are useful?

    plotSnapshotPerformance(resultClasses, "Snapshot Performance")
    # plotBestCurriculumResults(resultClasses, "Best Curriculum Results")
    # plotEpochAvgCurricReward(resultClasses, "Average Curriculum Reward of all Generations in an epoch")

