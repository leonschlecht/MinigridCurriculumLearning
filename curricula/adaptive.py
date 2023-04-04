import json
import os
from utils import ENV_NAMES
from scripts import train, evaluate


def adaptiveCurriculum(args, ITERATIONS_PER_ENV, txtLogger, startTime):
    """
    Trains on an adaptive curriculum, i.e. depending on the performance of the agent, the next envs will be determined
    """

    modelPath = os.getcwd() + "\\storage\\" + args.model
    logFilePath = modelPath + "\\status.json"

    if os.path.exists(logFilePath):
        with open(logFilePath, 'r') as f:
            trainingInfoJson = json.loads(f.read())

        iterationsDoneSoFar = trainingInfoJson["numFrames"]
        startEpoch = trainingInfoJson["epochsDone"]
        rewards = trainingInfoJson["rewards"]
        easierEnv = calculateNextEnvs(rewards[-1])
        txtLogger.info(f"Continung training from epoch {startEpoch}... ")
    else:
        startEpoch = 0
        iterationsDoneSoFar = 0
        easierEnv = ENV_NAMES.DOORKEY_6x6
        trainingInfoJson = {
            "curriculaEnvs": [],  # list of lists
            "rewards": [],
            "epochsDone": 0,
            "numFrames": 0}
        with open(logFilePath, 'w') as f:
            f.write(json.dumps(trainingInfoJson, indent=4))
    harderEnv = nextEnv(easierEnv)
    for epoch in range(startEpoch, 25):
        curriculum = [harderEnv, easierEnv, harderEnv, easierEnv]
        for env in curriculum:
            iterationsDoneSoFar = train.main(iterationsDoneSoFar + ITERATIONS_PER_ENV, args.model, env, args)

        evaluationScore = evaluate.evaluateAgent(args.model, args)
        easierEnv, harderEnv = calculateNextEnvs(evaluationScore)
        trainingInfoJson["curriculaEnvs"].append([curriculum])  # list of lists
        trainingInfoJson["rewards"].append(evaluationScore)
        trainingInfoJson["epochsDone"] = epoch
        trainingInfoJson["numFrames"] = iterationsDoneSoFar
        txtLogger.info(
            f"evaluationScore in ep {epoch}: {evaluationScore} ---> next Env: {easierEnv}; iterations: {iterationsDoneSoFar}")
        with open(logFilePath, 'w') as f:
            f.write(json.dumps(trainingInfoJson, indent=4))
    print("Done")


def calculateNextEnvs(score):
    if score <= 1:
        easierEnv = ENV_NAMES.DOORKEY_5x5
    elif score <= 2:
        easierEnv = ENV_NAMES.DOORKEY_6x6
    elif score <= 3:
        easierEnv = ENV_NAMES.DOORKEY_8x8
    else:
        easierEnv = ENV_NAMES.DOORKEY_16x16
    return easierEnv, nextEnv(easierEnv)


def nextEnv(currentEnv):
    if currentEnv == ENV_NAMES.DOORKEY_8x8:
        return ENV_NAMES.DOORKEY_16x16
    if currentEnv == ENV_NAMES.DOORKEY_6x6:
        return ENV_NAMES.DOORKEY_8x8
    if currentEnv == ENV_NAMES.DOORKEY_5x5:
        return ENV_NAMES.DOORKEY_6x6
    return ENV_NAMES.DOORKEY_16x16


def prevEnv(currentEnv):
    if currentEnv == ENV_NAMES.DOORKEY_16x16:
        return ENV_NAMES.DOORKEY_8x8
    if currentEnv == ENV_NAMES.DOORKEY_8x8:
        return ENV_NAMES.DOORKEY_6x6
    if currentEnv == ENV_NAMES.DOORKEY_6x6:
        return ENV_NAMES.DOORKEY_5x5
    return ENV_NAMES.DOORKEY_5x5

def evaluateAdaptiveCurriculum(trainingInfoJson):
    print(123)