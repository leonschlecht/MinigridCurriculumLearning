import utils
from curricula import RollingHorizonEvolutionaryAlgorithm
from utils import initializeArgParser, ENV_NAMES
import os
from utils.curriculumHelper import *
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict


def newLines():
    print("\n----------------\n")


def plotPerformance(y: list[float], maxReward: int, modelName: str, iterationsPerEnv: int):
    x = range(1, len(y) + 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()  # create a second x-axis
    ax1.plot(x, y)
    ax1.scatter(x, y, marker='x', color='black')
    ax1.set_xticks(x)
    ax1.set_ylim(0, maxReward * 1.01)
    ax1.axhline(y=maxReward, color='red')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('reward')
    ax1.set_title(f'performance after each epoch (model:{modelName})')

    # set limits and ticks for the second x-axis
    ax2.set_xlim(ax1.get_xlim())
    new_tick_locations = [(i * 2) * iterationsPerEnv for i in range(1, len(y) // 2)]
    ax2.set_xticks(new_tick_locations)
    ax2.set_xticklabels([str(tick // 1000) + 'k' for tick in new_tick_locations])
    ax2.set_xlabel('#iterations')
    plt.show()


def plotSnapshotPerformance(y: list, stepMaxReward: int, modelName: str, iterationsPerEnv: int):
    plotPerformance(y, stepMaxReward, modelName, iterationsPerEnv)


def plotEpochAvgCurricReward(y: list, stepMaxReward: int, modelName: str, iterationsPerEnv: int):
    plotPerformance(y, stepMaxReward, modelName, iterationsPerEnv)


def plotBestCurriculumResults(y: list, curricMaxReward: int, modelName: str, iterationsPerEnv: int):
    plotSnapshotPerformance(y, curricMaxReward, modelName, iterationsPerEnv)


def plotEnvsUsedDistribution(envDistribution: dict, titleInfo='...'):
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
    title = 'Distribution of selected envs in ' +  titleInfo
    ax.set_title(title)
    ax.set_ylim(0, max(envOccurrences) * 1.1)

    # Add labels to the bars
    ax.bar_label(bar_container, labels=envOccurrences, fontsize=12, padding=5)

    plt.show()


def evaluateCurriculumResults(evaluationDictionary):
    selectedEnvList = evaluationDictionary[selectedEnvs]
    epochsTrained = evaluationDictionary[epochsDone] - 1
    framesTrained = evaluationDictionary[numFrames]
    modelPerformance = evaluationDictionary[actualPerformance]  # {"curricScoreRaw", "curricScoreNormalized", "snapshotScoreRaw", "curriculum"}
    keyList = evaluationDictionary.keys()
    if snapshotScoreKey in keyList:
        snapshotScores = evaluationDictionary[snapshotScoreKey]
    else:
        snapshotScores = []
        for epochDict in modelPerformance:
            snapshotScores.append(epochDict[snapshotScoreKey])

    bestCurriculaDict = evaluationDictionary[bestCurriculas]

    rewardsDict = evaluationDictionary[rewardsKey]

    stepMaxReward = evaluationDictionary[maxStepRewardKey]
    curricMaxReward = evaluationDictionary[maxCurricRewardKey]

    fullEnvDict = evaluationDictionary[curriculaEnvDetailsKey]
    difficultyList = evaluationDictionary[difficultyKey]
    trainingTimeList = evaluationDictionary[epochTrainingTime]
    trainingTimeSum = evaluationDictionary[sumTrainingTime]

    for epochKey in rewardsDict:
        epochDict = rewardsDict[epochKey]
        for genKey in epochDict:
            genRewardsStr = epochDict[genKey]
            numbers = re.findall(r'[0-9]*\.?[0-9]+', genRewardsStr)
            numbers = [float(n) for n in numbers]
            epochDict[genKey] = numbers
        # TODO assertion to make sure length hasnt chagned

    argsString: str = trainingInfoDict[fullArgs]
    loadedArgsDict: dict = {k.replace('Namespace(', ''): v for k, v in [pair.split('=') for pair in argsString.split(', ')]}

    modelName = loadedArgsDict[argsModelKey]

    if iterationsPerEnvKey in trainingInfoDict.keys():
        iterationsPerEnv = int(trainingInfoDict[iterationsPerEnvKey])
    else:
        iterationsPerEnv = int(loadedArgsDict[oldArgsIterPerEnvName])  # TODO this might become deprecated if I change iterPerEnv -> stepsPerEnv

    curricScores = []
    avgEpochRewards = []
    numCurric = float(loadedArgsDict[numCurricKey])
    newLines()
    i = 0
    if loadedArgsDict[trainEvolutionary]:
        for epochKey in rewardsDict:
            epochDict = rewardsDict[epochKey]
            genNr, listIdx = RollingHorizonEvolutionaryAlgorithm.getGenAndIdxOfBestIndividual(epochDict)
            bestCurricScore = epochDict[GEN_PREFIX + genNr][listIdx]
            curricScores.append(bestCurricScore)
            epochRewardsList = np.array(list(epochDict.values()))
            avgEpochRewards.append(np.sum(epochRewardsList))
            i += 1
        noOfGens: float = float(loadedArgsDict[nGenerations])
        maxCurricAvgReward = curricMaxReward * noOfGens * numCurric

    assert type(iterationsPerEnv) == int
    assert epochsTrained == len(rewardsDict.keys())

    # plotSnapshotPerformance(snapshotScores, stepMaxReward, modelName, iterationsPerEnv)
    # plotBestCurriculumResults(curricScores, curricMaxReward, modelName, iterationsPerEnv)
    # plotEpochAvgCurricReward(avgEpochRewards, maxCurricAvgReward, modelName, iterationsPerEnv)

    # TODO plot the envs used
    # TODO also for selectedEnvs
    usedEnvEnumeration = trainingInfoDict[usedEnvEnumerationKey]
    snapshotEnvDistribution = {env: 0 for env in usedEnvEnumeration}
    for curricStep in selectedEnvList:
        for env in curricStep:
            envStrRaw = env.split("-custom")[0]
            snapshotEnvDistribution[envStrRaw] += 1

    bestCurriculaEnvDistribution = {env: 0 for env in usedEnvEnumeration}

    for epochList in bestCurriculaDict:
        for curricStep in epochList:
            for env in curricStep: # TODO maybe this might break because of recent log changes
                envStrRaw = env.split("-custom")[0]
                bestCurriculaEnvDistribution[envStrRaw] += 1

    allCurricDistribution = {env: 0 for env in usedEnvEnumeration}
    for epochKey in fullEnvDict:
        epochGenDict = fullEnvDict[epochKey]
        for genKey in epochGenDict:
            curriculumList = epochGenDict[genKey]
            for curric in curriculumList:
                for curricStep in curric:
                    for env in curricStep:
                        envStrRaw = env.split("-custom")[0]
                        allCurricDistribution[envStrRaw] += 1

    plotEnvsUsedDistribution(allCurricDistribution)

    plt.show()
    # TODO plot the snapshot vs curricReward problem
    # TODO plot reward development of 1 curriculum over multiple generations
    # TODO find out a way to properly plot the difficulty list / maybe how it influences the results; and maybe how you can improve it so that it is not even needed in the first place
    # TODO find way to plot multiple models at once (and show some relevant legend for info of model name or sth like that)

    # TODO somehow reference the epochs associated with certain rewards
    # TODO log avg first step reward



if __name__ == "__main__":
    args = initializeArgParser()  # TODO there should be a slimmer version of argparse for this case
    logFilePath = os.getcwd() + "\\storage\\" + args.model + "\\status.json"

    if os.path.exists(logFilePath):
        with open(logFilePath, 'r') as f:
            trainingInfoDict = json.loads(f.read())
        assert trainingInfoDict is not None
        evaluateCurriculumResults(trainingInfoDict)

    else:
        raise Exception("Model doesnt exist!")
    # Given a model name (which should probably have the trained method in it as well)
