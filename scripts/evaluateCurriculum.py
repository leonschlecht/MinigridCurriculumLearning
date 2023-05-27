import utils
from curricula import RollingHorizonEvolutionaryAlgorithm
from utils import initializeArgParser, ENV_NAMES
import os
from utils.curriculumHelper import *
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def tmp():
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


def plotEnvsUsedDistribution(envDistribution: dict):
    d = {'MiniGrid-DoorKey-16x16': 0, 'MiniGrid-DoorKey-8x8': 8, 'MiniGrid-DoorKey-6x6': 13, 'MiniGrid-DoorKey-5x5': 1}
    original_dict = {'MiniGrid-DoorKey-16x16': 0, 'MiniGrid-DoorKey-8x8': 8, 'MiniGrid-DoorKey-6x6': 13, 'MiniGrid-DoorKey-5x5': 1}

    # Step 1: Extract the numeric part of the string from the keys of the dictionary
    new_dict = {}
    for numKey in original_dict.keys():
        numeric_part = numKey.split('-')[-1]
        new_dict[numeric_part] = original_dict[numKey]

    keyValues = []
    for fullKey in new_dict:
        numKey = int(fullKey.split("x")[0])
        keyValues.append(numKey)
    keyValues = sorted(keyValues)

    tmp = []
    for numKey in keyValues:
        for dictKey in new_dict:
            if str(numKey) == dictKey.split('x')[0]:
                tmp.append((dictKey, numKey))
                break

    sortedKeys = [fullKey for fullKey, _ in tmp]
    finalDict = {k: new_dict[k] for k in sortedKeys}

    print(finalDict)

    keys = list(finalDict.keys())
    values = list(finalDict.values())

    fig, ax = plt.subplots()
    bar_container = ax.bar(keys, values)

    ax.set_ylabel('Count')
    ax.set_title('Distribution of Keys')
    ax.set_ylim(0, max(values) * 1.1)

    # Add labels to the bars
    ax.bar_label(bar_container, labels=values, fontsize=12, padding=5)

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
    tmp()
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
    envDistribution = {env: 0 for env in usedEnvEnumeration}
    for envsInStep in selectedEnvList:
        for env in envsInStep:
            envStrRaw = env.split("-custom")[0]
            envDistribution[envStrRaw] += 1

    plotEnvsUsedDistribution(envDistribution)

    plt.show()
    # TODO plot the snapshot vs curricReward problem
    # TODO plot reward development of 1 curriculum over multiple generations
    # TODO find out a way to properly plot the difficulty list / maybe how it influences the results; and maybe how you can improve it so that it is not even needed in the first place
    # TODO find way to plot multiple models at once (and show some relevant legend for info of model name or sth like that)

    # TODO somehow reference the epochs associated with certain rewards
    # TODO log avg first step reward

    fullEnvList = evaluationDictionary[curriculaEnvDetailsKey]
    difficultyList = evaluationDictionary[difficultyKey]
    trainingTimeList = evaluationDictionary[epochTrainingTime]
    trainingTimeSum = evaluationDictionary[sumTrainingTime]


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
