from curricula import RollingHorizonEvolutionaryAlgorithm
from utils import initializeArgParser
import os
from utils.curriculumHelper import *
import matplotlib.pyplot as plt
import numpy as np

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
    title = 'Distribution of selected envs in ' + titleInfo
    ax.set_title(title)
    ax.set_ylim(0, max(envOccurrences) * 1.1)

    # Add labels to the bars
    ax.bar_label(bar_container, labels=envOccurrences, fontsize=12, padding=5)

    plt.show()


def getSnapshotScores(evalDict: dict, modelPerformance: dict) -> list[float]:
    """
    Gets the 1st step scores of the model and returns it as a list.
    :param evalDict:
    :param modelPerformance:
    :return:
    """
    if snapshotScoreKey in evalDict.keys():
        snapshotScores = evalDict[snapshotScoreKey]
    else:
        snapshotScores = []
        for epochDict in modelPerformance:
            snapshotScores.append(epochDict[snapshotScoreKey])
    return snapshotScores


def getEpochDict(rewardsDict):
    """
    Transforms the str list of the rewards dictionary into a dictionary that has the lists with float values instead of str values
    """
    epochDict = {}
    for epochKey in rewardsDict:
        epochDict = rewardsDict[epochKey]
        for genKey in epochDict:
            genRewardsStr = epochDict[genKey]
            numbers = re.findall(r'[0-9]*\.?[0-9]+', genRewardsStr)
            numbers = [float(n) for n in numbers]
            epochDict[genKey] = numbers
        # TODO assertion to make sure length hasnt chagned
    return epochDict


def getBestCurriculaEnvDistribution(bestCurriculaDict, usedEnvEnumeration, ):
    bestCurriculaEnvDistribution = {env: 0 for env in usedEnvEnumeration}
    for epochList in bestCurriculaDict:
        for curricStep in epochList:
            for env in curricStep:  # TODO this might break because of recent log changes -> test it
                envStrRaw = env.split("-custom")[0]
                bestCurriculaEnvDistribution[envStrRaw] += 1
    return bestCurriculaEnvDistribution


def getSnapshotEnvDistribution(selectedEnvList, usedEnvEnumeration):
    snapshotEnvDistribution = {env: 0 for env in usedEnvEnumeration}
    for curricStep in selectedEnvList:
        for env in curricStep:
            envStrRaw = env.split("-custom")[0]
            snapshotEnvDistribution[envStrRaw] += 1
    return snapshotEnvDistribution


def getAllCurriculaEnvDistribution(fullEnvDict, usedEnvEnumeration):
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
    return allCurricDistribution


def getIterationsPerEnv(trainingInfoDict, loadedArgsDict):
    if iterationsPerEnvKey in trainingInfoDict.keys():
        iterationsPerEnv = int(trainingInfoDict[iterationsPerEnvKey])
    else:
        iterationsPerEnv = int(loadedArgsDict[oldArgsIterPerEnvName])  # TODO this might become deprecated if I change iterPerEnv -> stepsPerEnv

    return iterationsPerEnv


def evaluateCurriculumResults(evaluationDictionary):
    argsString: str = trainingInfoDictionary[fullArgs]
    loadedArgsDict: dict = {k.replace('Namespace(', ''): v for k, v in [pair.split('=') for pair in argsString.split(', ')]}

    selectedEnvList = evaluationDictionary[selectedEnvs]
    epochsTrained = evaluationDictionary[epochsDone] - 1
    framesTrained = evaluationDictionary[numFrames]
    modelPerformance = evaluationDictionary[actualPerformance]  # {"curricScoreRaw", "curricScoreNormalized", "snapshotScoreRaw", "curriculum"}
    snapShotScores = getSnapshotScores(evaluationDictionary, modelPerformance)

    bestCurriculaDict = evaluationDictionary[bestCurriculas]
    rewardsDict = evaluationDictionary[rewardsKey]

    stepMaxReward = evaluationDictionary[maxStepRewardKey]
    curricMaxReward = evaluationDictionary[maxCurricRewardKey]

    fullEnvDict = evaluationDictionary[curriculaEnvDetailsKey]
    difficultyList = evaluationDictionary[difficultyKey]
    trainingTimeList = evaluationDictionary[epochTrainingTime]
    trainingTimeSum = evaluationDictionary[sumTrainingTime]

    epochDict = getEpochDict(rewardsDict)

    modelName = loadedArgsDict[argsModelKey]
    iterationsPerEnv = getIterationsPerEnv(trainingInfoDictionary, loadedArgsDict)

    curricScores = []
    avgEpochRewards = []
    numCurric = float(loadedArgsDict[numCurricKey])
    if loadedArgsDict[trainEvolutionary]:
        for epochKey in rewardsDict:
            epochDict = rewardsDict[epochKey]
            genNr, listIdx = RollingHorizonEvolutionaryAlgorithm.getGenAndIdxOfBestIndividual(epochDict)
            bestCurricScore = epochDict[GEN_PREFIX + genNr][listIdx]
            curricScores.append(bestCurricScore)
            epochRewardsList = np.array(list(epochDict.values()))
            avgEpochRewards.append(np.sum(epochRewardsList))

        noOfGens: float = float(loadedArgsDict[nGenerations])
        maxCurricAvgReward = curricMaxReward * noOfGens * numCurric

    assert type(iterationsPerEnv) == int
    assert epochsTrained == len(rewardsDict.keys())

    usedEnvEnumeration = trainingInfoDictionary[usedEnvEnumerationKey]
    snapshotEnvDistribution = getSnapshotEnvDistribution(selectedEnvList, usedEnvEnumeration)
    bestCurriculaEnvDistribution = getBestCurriculaEnvDistribution(bestCurriculaDict, usedEnvEnumeration)
    allCurricDistribution = getAllCurriculaEnvDistribution(fullEnvDict, usedEnvEnumeration)

    # plotEnvsUsedDistribution(allCurricDistribution, "all Curric Distribution")
    # plotEnvsUsedDistribution(snapshotEnvDistribution, "snapshot Distribution")
    # plotEnvsUsedDistribution(bestCurriculaEnvDistribution, "best Curricula Distribution")
    # plotSnapshotPerformance(snapshotScores, stepMaxReward, modelName, iterationsPerEnv)
    # plotBestCurriculumResults(curricScores, curricMaxReward, modelName, iterationsPerEnv)
    # plotEpochAvgCurricReward(avgEpochRewards, maxCurricAvgReward, modelName, iterationsPerEnv)

    # TODO plot the snapshot vs curricReward problem
    # TODO plot reward development of 1 curriculum over multiple generations
    # TODO find out a way to properly plot the difficulty list / maybe how it influences the results; and maybe how you can improve it so that it is not even needed in the first place
    # TODO find way to plot multiple models at once (and show some relevant legend for info of model name or sth like that)
    # TODO save the plots


if __name__ == "__main__":
    evalDirectory = os.getcwd() + "\\storage\\save\\evaluate\\"
    logFilePaths = []
    evalDirectories = next(os.walk(evalDirectory))[1]
    for model in evalDirectories:
        logFilePaths.append(evalDirectory + model + "\\status.json")

    modelDictionaries = []
    for logFilePath in logFilePaths:
        if os.path.exists(logFilePath):
            with open(logFilePath, 'r') as f:
                trainingInfoDictionary = json.loads(f.read())
            assert trainingInfoDictionary is not None
            modelDictionaries.append(trainingInfoDictionary)
        else:
            raise Exception(f"Path '{logFilePath}' doesnt exist!")
    print(len(modelDictionaries))
    # evaluateCurriculumResults(trainingInfoDictionary)

