import utils
from utils import initializeArgParser
import os
import json
from utils.curriculumHelper import *


def evaluateCurriculumResults(evaluationDictionary, isRHEA=False):
    # evaluationDictionary["actualPerformance"][0] ---> zeigt den avg reward des models zu jedem übernommenen Snapshot
    # evaluationDictionary["actualPerformance"][1] ---> zeigt die zuletzt benutzte Umgebung zu dem Zeitpunkt an

    selectedEnvList = evaluationDictionary[selectedEnvs]
    epochsTrained = evaluationDictionary[epochsDone]
    framesTrained = evaluationDictionary[numFrames]
    modelPerformance = evaluationDictionary[actualPerformance]# {"curricScoreRaw", "curricScoreNormalized", "snapshotScoreRaw", "curriculum"}

    try:
        snapshotScores = evaluationDictionary[snapshotScoreKey] # if there is an error here, hopefully the values are stored in the actualPerformance part
    except KeyError:
        print(modelPerformance)
        snapshotScores = []
        print(modelPerformance)
        print("---------")
        for epochDict in modelPerformance:
            snapshotScores.append(epochDict[snapshotScoreKey])

    bestCurriculaDict = evaluationDictionary[bestCurriculas]

    rewardsDict = evaluationDictionary[rewardsKey]
    for x in rewardsDict:
        epochDict = rewardsDict[x]
        for gen in epochDict:
            genRewardsStr = epochDict[gen]
            numbers = re.findall(r'\d+\.\d+', genRewardsStr)
            numbers = list(map(float, numbers))
            epochDict[gen] = numbers
    print(rewardsDict)
    # TODO copy get best gen / individual stuff from RHEA train
    # TODO: plot the snapshotscores (copy from colab)
    # TODO: plot maxCurricReward of each epoch
    # TODO plot the snapshot vs curricReward problem
    # plot reward development of 1 curriculum over multiple generations
    # plot avg reward of each generation of an epoch
    # plot max Reward of each gen

    exit()

    fullEnvList = evaluationDictionary[curriculaEnvDetailsKey]
    difficultyList = evaluationDictionary[difficultyKey]
    trainingTimeList = evaluationDictionary[epochTrainingTime]
    trainingTimeSum = evaluationDictionary[sumTrainingTime]


if __name__ == "__main__":
    args = initializeArgParser()
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

