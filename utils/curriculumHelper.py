from datetime import datetime
import json

from utils import ENV_NAMES, getEnvFromDifficulty
import random

###### DEFINE CONSTANTS AND DICTIONARY KEYS #####

GEN_PREFIX = 'gen'

selectedEnvs = "selectedEnvs"
bestCurriculas = "bestCurriculas"
curriculaEnvDetails = "curriculaEnvDetails"
rewardsKey = "rewards"
actualPerformance = "actualPerformance"
epochsDone = "epochsDone"
numFrames = "numFrames"
cmdLineStringKey = "cmdLineString"
epochTrainingTime = "epochTrainingTime"
sumTrainingTime = "sumTrainingTime"
difficultyKey = "difficultyKey"


def evaluateCurriculumResults(evaluationDictionary):
    # evaluationDictionary["actualPerformance"][0] ---> zeigt den avg reward des models zu jedem übernommenen Snapshot
    # evaluationDictionary["actualPerformance"][1] ---> zeigt die zuletzt benutzte Umgebung zu dem Zeitpunkt an
    #
    tmp = []
    i = 0
    for reward, env in tmp:
        print(reward, env)
        i += 1

    # Dann wollen wir sehen, wie das curriculum zu dem jeweiligen zeitpunkt ausgesehen hat.
    # # Aber warum? Und wie will man das nach 20+ durchläufen plotten


def saveTrainingInfoToFile(path, jsonBody):
    with open(path, 'w') as f:
        f.write(json.dumps(jsonBody, indent=4, default=str))


def printFinalLogs(trainingInfoJson, txtLogger) -> None:
    """
    Prints the last logs, after the training is done
    """
    txtLogger.info("----TRAINING END-----")
    txtLogger.info(f"Best Curricula {trainingInfoJson[bestCurriculas]}")
    txtLogger.info(f"Trained in Envs {trainingInfoJson[selectedEnvs]}")
    txtLogger.info(f"Rewards: {trainingInfoJson[rewardsKey]}")

    now = datetime.now()
    timeDiff = 0
    print(timeDiff)
    txtLogger.info(f"Time ended at {now} , total training time: {timeDiff}")
    txtLogger.info("-------------------\n\n")


def initTrainingInfo(cmdLineString, logFilePath) -> dict:
    """
    Initializes the trainingInfo dictionary
    :return:
    """
    trainingInfoJson = {selectedEnvs: [],
                        bestCurriculas: [],
                        curriculaEnvDetails: {},
                        rewardsKey: {},
                        actualPerformance: [],
                        epochsDone: 1,
                        epochTrainingTime: [],
                        sumTrainingTime: 0,
                        cmdLineStringKey: cmdLineString,
                        difficultyKey: [0],
                        numFrames: 0}
    saveTrainingInfoToFile(logFilePath, trainingInfoJson)
    return trainingInfoJson


def logInfoAfterEpoch(epoch, currentBestCurriculum, currentReward, trainingInfoJson, txtLogger, maxReward,
                      totalEpochs):
    """
    Logs relevant training info after a training epoch is done and the trainingInfo was updated
    :param totalEpochs:
    :param epoch:
    :param currentBestCurriculum: the id of the current best curriculum
    :param currentReward:
    :return:
    """
    selectedEnv = trainingInfoJson[selectedEnvs][-1]

    txtLogger.info(
        f"Best results in epoch {epoch} came from curriculum {currentBestCurriculum}")
    txtLogger.info(
        f"CurriculaEnvDetails {curriculaEnvDetails}; selectedEnv: {selectedEnv}")
    txtLogger.info(f"Current Reward: {currentReward}. That is {currentReward / maxReward} of maxReward")

    txtLogger.info(f"\nEPOCH: {epoch} SUCCESS (total: {totalEpochs})\n ")


def calculateEnvDifficulty(currentReward, maxReward) -> int:
    # TODO EXPERIMENT: that is why i probably should have saved the snapshot reward
    if currentReward < maxReward * .25:
        return 0
    elif currentReward < maxReward * .75:
        return 1
    return 2


def randomlyInitializeCurricula(numberOfCurricula: int, envsPerCurriculum: int, envDifficulty: int) -> list:
    """
    Initializes list of curricula randomly
    :param envDifficulty:
    :param numberOfCurricula: how many curricula will be generated
    :param envsPerCurriculum: how many environment each curriculum has
    """
    curricula = []
    for i in range(numberOfCurricula):
        indices = random.sample(range(len(ENV_NAMES.ALL_ENVS)), envsPerCurriculum)
        newCurriculum = [getEnvFromDifficulty(idx, envDifficulty) for idx in indices]
        curricula.append(newCurriculum)
    assert len(curricula) == numberOfCurricula
    return curricula


def updateTrainingInfo(trainingInfoJson, epoch: int, bestCurriculum: list, currentRewards, currentScore: float,
                       iterationsDone, envDifficulty: int, lastEpochStartTime, curricula, logFilePath, popX=None) -> None:
    """
    Updates the training info dictionary
    :param logFilePath:
    :param curricula:
    :param lastEpochStartTime:
    :param envDifficulty:
    :param iterationsDone:
    :param trainingInfoJson:
    :param epoch: current epoch
    :param bestCurriculum: the curriculum that had the highest reward in the latest epoch
    :param currentRewards: the dict of rewards for each generation and each curriculum
    :param currentScore: the current best score
    :param popX: the pymoo X parameter for debugging purposes - only relevant for RHEA, not RRH
    """
    trainingInfoJson[epochsDone] = epoch + 1
    trainingInfoJson[numFrames] = iterationsDone

    trainingInfoJson[selectedEnvs].append(bestCurriculum[0])
    trainingInfoJson[bestCurriculas].append(bestCurriculum)
    trainingInfoJson[rewardsKey] = currentRewards  # TODO test this ? or does this not overwrite everything
    trainingInfoJson[actualPerformance].append([currentScore, bestCurriculum])
    trainingInfoJson[curriculaEnvDetails]["epoch" + str(epoch)] = curriculaEnvDetails
    trainingInfoJson[difficultyKey].append(envDifficulty)

    now = datetime.now()
    timeSinceLastEpoch = (now - lastEpochStartTime).total_seconds()
    trainingInfoJson[epochTrainingTime].append(timeSinceLastEpoch)
    trainingInfoJson[sumTrainingTime] += timeSinceLastEpoch

    # Debug Logs
    trainingInfoJson["currentListOfCurricula"] = curricula  # TODO is this useful?
    if popX is not None:
        trainingInfoJson["curriculumListAsX"] = popX

    saveTrainingInfoToFile(logFilePath, trainingInfoJson)
    # TODO how expensive is it to always overwrite everything?
