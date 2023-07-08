import json
import re
from datetime import datetime

import numpy as np
from gymnasium.envs.registration import register

from utils import ENV_NAMES

###### DEFINE CONSTANTS AND DICTIONARY KEYS #####

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
nGenerations = "nGen"
numCurricKey = "numCurric"
usedEnvEnumerationKey = "usedEnvEnumeration"
modelKey = "model"

# Evaluation Keys
snapshotDistributionKey = "snapshotDistribution"
bestCurricDistributionKey = "bestCurricDistribution"
allCurricDistributoinKey = "allCurricDistribution"

# Used for all Paralell training
NEXT_ENVS = "NextEnvs"


def saveTrainingInfoToFile(path, jsonBody):
    with open(path, 'w') as f:
        f.write(json.dumps(jsonBody, indent=4, default=str))


def printFinalLogs(trainingInfoJson, txtLogger) -> None:
    """
    Prints the last logs, after the training is done
    """
    txtLogger.info("\n\n\n----TRAINING END-----")
    txtLogger.info(f"Num Frames {trainingInfoJson[numFrames]}")
    now = datetime.now()
    txtLogger.info(f"Time ended at {now} , total training time: {trainingInfoJson[sumTrainingTime]}")
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


def calculateEnvDifficulty(iterationsDone, difficultyStepsize) -> float:
    startDecreaseNum = 500000
    if iterationsDone <= startDecreaseNum:
        value = 1
    else:
        value = 1 - ((iterationsDone - startDecreaseNum) / difficultyStepsize / 20)
    value = max(value, 0.15)

    assert value <= 1
    if value < 1:
        register(
            id=ENV_NAMES.DOORKEY_12x12 + ENV_NAMES.CUSTOM_POSTFIX + str(value),
            entry_point="minigrid.envs:DoorKeyEnv",
            kwargs={"size": 12, "max_steps": int(maxStepsEnv4 * value)},
        )
        register(
            id=ENV_NAMES.DOORKEY_10x10 + ENV_NAMES.CUSTOM_POSTFIX + str(value),
            entry_point="minigrid.envs:DoorKeyEnv",
            kwargs={"size": 10, "max_steps": int(maxStepsEnv4 * value)},
        )

        register(
            id=ENV_NAMES.DOORKEY_8x8 + ENV_NAMES.CUSTOM_POSTFIX + str(value),
            entry_point="minigrid.envs:DoorKeyEnv",
            kwargs={"size": 8, "max_steps": int(maxStepsEnv4 * value)},
        )

        register(
            id=ENV_NAMES.DOORKEY_6x6 + ENV_NAMES.CUSTOM_POSTFIX + str(value),
            entry_point="minigrid.envs:DoorKeyEnv",
            kwargs={"size": 6, "max_steps": int(maxStepsEnv4 * value)},
        )
    return value


ENV_SIZE_POWER = 2
SIZE_MUTIPLICATOR = 10
maxStepsEnv4 = 12 ** ENV_SIZE_POWER * SIZE_MUTIPLICATOR
maxStepsEnv3 = 10 ** ENV_SIZE_POWER * SIZE_MUTIPLICATOR
maxStepsEnv2 = 8 ** ENV_SIZE_POWER * SIZE_MUTIPLICATOR
maxStepsEnv1 = 6 ** ENV_SIZE_POWER * SIZE_MUTIPLICATOR
