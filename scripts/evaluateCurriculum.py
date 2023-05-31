import os

import matplotlib.pyplot as plt

from scripts.Result import Result
from utils.curriculumHelper import *


def plotPerformance(allLists: list[list[float]], maxReward: int, iterationsPerEnv: int, title: str):
    aList = allLists[0]
    # TODO IDK how useful iterationsPerEnv is here ?? important if it is the varied parameter
    # Maybe I need a pair of y: the values as they are, and x: the iterationSteps. So f(50k) = Reward, f(100k); so it is easier to comprae
    x = range(1, len(aList) + 1)
    # TODO what if maxReward are not all equal for all currics: maybe envorce normalized one
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()  # create a second x-axis
    ax1.plot(x, aList)
    ax1.scatter(x, aList, marker='x', color='black')
    ax1.set_xticks(x)
    ax1.set_ylim(0, maxReward * 1.01)
    ax1.axhline(y=maxReward, color='red')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('reward')
    ax1.set_title(title)

    # set limits and ticks for the second x-axis
    ax2.set_xlim(ax1.get_xlim())
    new_tick_locations = [(i * 2) * iterationsPerEnv for i in range(1, len(aList) // 2)]
    ax2.set_xticks(new_tick_locations)
    ax2.set_xticklabels([str(tick // 1000) + 'k' for tick in new_tick_locations])
    ax2.set_xlabel('#iterations')
    plt.show()


def plotSnapshotPerformance(results: list[Result], title: str):
    y = []
    for res in results:
        y.append(res.snapShotScores)
    maxReward = results[0].stepMaxReward
    iterationsPerEnv = results[0].iterationsPerEnv
    plotPerformance(y, maxReward, iterationsPerEnv, title)


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

    # plotSnapshotPerformance(resultClasses, "Snapshot Performance")
    # plotBestCurriculumResults(resultClasses, "Best Curriculum Results")
    plotEpochAvgCurricReward(resultClasses, "Average Curriculum Reward of all Generations in an epoch")
