import utils
from curricula import RollingHorizonEvolutionaryAlgorithm
from utils import initializeArgParser
import os
from utils.curriculumHelper import *
import matplotlib.pyplot as plt
import numpy as np


def tmp():
    print("\n----------------\n")


def plotSnapshotPerformance(y: list, stepMaxReward: int, modelName: str, iterationsPerEnv: int):
    x = range(1, len(y) + 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()  # create a second x-axis
    ax1.plot(x, y)
    ax1.scatter(x, y, marker='x', color='black')
    ax1.set_xticks(x)
    ax1.set_ylim(0, stepMaxReward * 1.01)
    ax1.axhline(y=stepMaxReward, color='red')
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


def plotBestCurriculumResults(y: list, curricMaxReward: int, modelName: str, iterationsPerEnv: int):
    x = range(1, len(y) + 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()  # create a second x-axis
    ax1.plot(x, y)
    ax1.scatter(x, y, marker='x', color='black')
    ax1.set_xticks(x)
    ax1.set_ylim(0, curricMaxReward * 1.01)
    ax1.axhline(y=curricMaxReward, color='red')
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


def evaluateCurriculumResults(evaluationDictionary, isRHEA=False):
    # evaluationDictionary["actualPerformance"][0] ---> zeigt den avg reward des models zu jedem übernommenen Snapshot
    # evaluationDictionary["actualPerformance"][1] ---> zeigt die zuletzt benutzte Umgebung zu dem Zeitpunkt an

    selectedEnvList = evaluationDictionary[selectedEnvs]
    epochsTrained = evaluationDictionary[epochsDone]
    framesTrained = evaluationDictionary[numFrames]
    modelPerformance = evaluationDictionary[actualPerformance]  # {"curricScoreRaw", "curricScoreNormalized", "snapshotScoreRaw", "curriculum"}
    keyList = evaluationDictionary.keys()
    if snapshotScoreKey in keyList:
        snapshotScores = evaluationDictionary[
            snapshotScoreKey]  # if there is an error here, hopefully the values are stored in the actualPerformance part
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
            numbers = re.findall(r'[0-9]*\.?[0-9]+', genRewardsStr)  # modified regex
            numbers = [float(n) for n in numbers]
            epochDict[genKey] = numbers
        # TODO assertion to make sure length hasnt chagned
    print(epochsTrained-1, len(rewardsDict.keys()))

    argsString: str = trainingInfoDict[fullArgs]
    loadedArgsDict: dict = {k.replace('Namespace(', ''): v for k, v in [pair.split('=') for pair in argsString.split(', ')]}

    modelName = loadedArgsDict[argsModelKey]

    if iterationsPerEnvKey in trainingInfoDict.keys():
        iterationsPerEnv = int(trainingInfoDict[iterationsPerEnvKey])
    else:
        iterationsPerEnv = int(loadedArgsDict[oldArgsIterPerEnvName])  # TODO this might become deprecated if I change iterPerEnv -> stepsPerEnv

    curricScores = []
    tmp()
    if loadedArgsDict[trainEvolutionary]:
        for epochKey in rewardsDict:
            epochDict = rewardsDict[epochKey]
            genNr, listIdx = RollingHorizonEvolutionaryAlgorithm.getGenAndIdxOfBestIndividual(epochDict)
            bestCurricScore = epochDict[GEN_PREFIX + genNr][listIdx]
            curricScores.append(bestCurricScore)


    assert type(iterationsPerEnv) == int

    # plotSnapshotPerformance(snapshotScores, stepMaxReward, modelName, iterationsPerEnv)
    plotBestCurriculumResults(curricScores, curricMaxReward, modelName, iterationsPerEnv)

    plt.show()
    # TODO copy get best gen / individual stuff from RHEA train
    # TODO: plot the snapshotscores (copy from colab)
    # TODO: plot maxCurricReward of each epoch
    # TODO plot the snapshot vs curricReward problem
    # TODO plot reward development of 1 curriculum over multiple generations
    # TODO plot avg reward of each generation of an epoch
    # TODO plot max Reward of each gen
    # TODO dont log "Device: ..." for loading here
    # TODO find out a way to properly plot the difficulty list / maybe how it influences the results; and maybe how you can improve it so that it is not even needed in the first place

    fullEnvList = evaluationDictionary[curriculaEnvDetailsKey]
    difficultyList = evaluationDictionary[difficultyKey]
    trainingTimeList = evaluationDictionary[epochTrainingTime]
    trainingTimeSum = evaluationDictionary[sumTrainingTime]


if __name__ == "__main__":
    args = initializeArgParser()  # TODO there should be a slimmer version of argparse for this case
    logFilePath = os.getcwd() + "\\storage\\" + args.model + "\\status.json"
    txtLogger = utils.get_txt_logger(utils.get_model_dir(args.model))

    if os.path.exists(logFilePath):
        with open(logFilePath, 'r') as f:
            trainingInfoDict = json.loads(f.read())
        assert trainingInfoDict is not None
        evaluateCurriculumResults(trainingInfoDict, isRHEA=True)

    else:
        print("Model doesnt exist!")
    # Given a model name (which should probably have the trained method in it as well)
