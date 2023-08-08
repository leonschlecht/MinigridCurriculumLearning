import json
import re
from datetime import datetime
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
rawRewardsKey = "rawRewards"
seedKey = "seed"
fullArgs = "args"
consecutivelyChosen = "consecutivelyChosen"
additionalNotes = "additionalNotes"
snapshotScoreKey = "snapshotScore"
iterationsPerEnvKey = "iterationsPerEnv"
maxStepRewardKey = "maxStepReward"
maxCurricRewardKey = "maxCurricReward"
iterationSteps = "iterationSteps"

MAX_REWARD_PER_ENV = 1

# Key names of hey they appear in the command line args
oldArgsIterPerEnvName = "iterPerEnv"
argsModelKey = "model"
trainEvolutionary = "trainEvolutionary"
trainRandomRH = "trainRandomRH"
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

# Evaluation font sizes
# TODO maybe use different file for this
labelFontsize = 16
titleFontsize = 20
legendFontSize = 14  # TODO why is this still so big sometimes ?
tickFontsize = 12


# Env sizes
def getDoorKeyMaxSteps(envSize: int) -> int:
    """
    Returns the maximum steps allowed for a given doorkey environment size
    """
    DOORKEY_MAXSTEP_MULTIPLICATOR = 10
    return envSize ** 2 * DOORKEY_MAXSTEP_MULTIPLICATOR


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


def calculateCurricStepMaxReward(allEnvs: list, noRewardShaping: bool) -> float:
    reward = 0
    for env in allEnvs:
        reward += getRewardMultiplier(env, noRewardShaping)
    maxReward: float = reward * MAX_REWARD_PER_ENV
    return maxReward


def calculateCurricMaxReward(curricLength, stepMaxReward, gamma) -> float:
    maxReward = 0
    for j in range(curricLength):
        maxReward += ((gamma ** j) * stepMaxReward)
    return maxReward


def getRewardMultiplier(evalEnv, noRewardShaping: bool):
    """

    :param evalEnv:
    :param noRewardShaping: whether or not to use rewardshaping (reward depend on env size or not)
    :return:
    """
    if noRewardShaping:
        return 1
    pattern = r'\d+'
    match = re.search(pattern, evalEnv)
    if match:
        return int(match.group())
    raise Exception("Something went wrong with the evaluation reward multiplier!", evalEnv)


def getDynamicObstacleMaxSteps(size):
    return 4 * size ** 2


def getNObstacles(size):
    return size // 2


def registerEnvs(selectedEnvsList: list, maxStepsPercent: float) -> None:
    """
    Register the new environments with the given updated max steps percentage
    :param selectedEnvsList:
    :param maxStepsPercent:
    :return:
    """
    for env in selectedEnvsList:
        size = int(env.split("-")[-1].split("x")[-1]) # DoorKey-5x5 ---> 5
        custom_postfix = ENV_NAMES.CUSTOM_POSTFIX + str(maxStepsPercent)

        if "DoorKey" in env:
            entry_point = "minigrid.envs:DoorKeyEnv"
            max_steps = int(getDoorKeyMaxSteps(size) * maxStepsPercent)
            kwargs = {"size": size, "max_steps": max_steps}
        elif "Dynamic-Obstacle" in env:
            entry_point = "minigrid.envs:DynamicObstaclesEnv"
            max_steps = int(getDynamicObstacleMaxSteps(size) * maxStepsPercent)
            kwargs = {"size": size, "n_obstacles": getNObstacles(size), "max_steps": max_steps}
            # TODO maybe add "agent_start_pos": None for random
        else:
            raise Exception("Env not found")

        register(
            id=env + custom_postfix,
            entry_point=entry_point,
            kwargs=kwargs,
        )
        print(env+custom_postfix)


def calculateEnvDifficulty(iterationsDone, difficultyStepsize, selectedEnvsList) -> float:
    startDecreaseNum = 500000
    if iterationsDone <= startDecreaseNum:
        newMaxStepsPercent: float = 1.0
    else:
        newMaxStepsPercent = 1 - ((iterationsDone - startDecreaseNum) / difficultyStepsize / 20)
    newMaxStepsPercent = max(newMaxStepsPercent, 0.15)

    assert newMaxStepsPercent <= 1
    if newMaxStepsPercent < 1:
        registerEnvs(selectedEnvsList, newMaxStepsPercent)

    return newMaxStepsPercent
