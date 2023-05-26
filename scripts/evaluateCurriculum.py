import utils
from curricula import RollingHorizonEvolutionaryAlgorithm
from utils import initializeArgParser
import os
from utils.curriculumHelper import *
import matplotlib.pyplot as plt
import numpy as np


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
    plotBestCurriculumResults(curricScores, curricMaxReward, modelName, iterationsPerEnv)
    plotEpochAvgCurricReward(avgEpochRewards, maxCurricAvgReward, modelName, iterationsPerEnv)

    plt.show()
    # TODO plot the snapshot vs curricReward problem
    # TODO plot reward development of 1 curriculum over multiple generations
    # TODO find out a way to properly plot the difficulty list / maybe how it influences the results; and maybe how you can improve it so that it is not even needed in the first place
    # TODO find way to plot multiple models at once (and show some relevant legend for info of model name or sth like that)
    # TODO plot the envs used

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
        print("Model doesnt exist!")
    # Given a model name (which should probably have the trained method in it as well)
