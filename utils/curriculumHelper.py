###### DEFINE CONSTANTS AND DICTIONARY KEYS #####
import json
import re
from datetime import datetime

GEN_PREFIX = 'gen'

# Dictionary keys
selectedEnvs = "selectedEnvs"
bestCurriculas = "bestCurriculas"
curriculaEnvDetailsKey = "curriculaEnvDetails"
rewardsKey = "curriculumRewards"
actualPerformance = "actualPerformance"
epochsDone = "epochsDone"
numFrames = "numFrames"
cmdLineStringKey = "cmdLineString"
epochTrainingTime = "epochTrainingTime"
sumTrainingTime = "sumTrainingTime"
difficultyKey = "difficultyKey"
seedKey = "seed"
fullArgs = "args"
consecutivelyChosen = "consecutivelyChosen"
additionalNotes = "additionalNotes"
snapshotScoreKey = "snapshotScore"
iterationsPerEnvKey = "iterationsPerEnv"
maxStepRewardKey = "maxStepReward"
maxCurricRewardKey = "maxCurricReward"

MAX_REWARD_PER_ENV = 1


# Key names of hey they appear in the command line args
oldArgsIterPerEnvName = "iterPerEnv"
argsModelKey = "model"
trainEvolutionary = "trainEvolutionary"
trainLinear = "trainLinear"
trainAdaptive = "trainAdaptive"
trainRandomRH = "trainRandomRH"
trainBiasedRandomRH = "trainBiasedRandomRH"
trainAllParalell = "trainAllParalell"


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


def calculateCurricStepMaxReward(allEnvs: list) -> float:
    reward = 0
    for env in allEnvs:
        reward += getRewardMultiplier(env)
    maxReward: float = reward * MAX_REWARD_PER_ENV
    return maxReward


def calculateCurricMaxReward(curricLength, stepMaxReward, gamma) -> float:
    maxReward = 0
    for j in range(curricLength):
        maxReward += ((gamma ** j) * stepMaxReward)
    return maxReward

def getRewardMultiplier(evalEnv):
    """

    :param evalEnv:
    :return:
    """
    pattern = r'\d+'
    match = re.search(pattern, evalEnv)
    if match:
        return int(match.group())
    raise Exception("Something went wrong with the evaluation reward multiplier!", evalEnv)
