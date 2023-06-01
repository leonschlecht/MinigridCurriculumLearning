from curricula import train, evaluate
from utils import ENV_NAMES
import time
import os
import json

"""
Trains each curriculum in order for a pre-set amount of iterations
"""


def startLinearCurriculum(txtLogger, startTime, args):
    TOTAL_ITERATIONS = 20000000
    iterations5x5 = 500000
    iterations6x6 = 500000
    iterations8x8 = 4000000
    iterations16x16 = TOTAL_ITERATIONS - iterations8x8 - iterations6x6 - iterations5x5
    curriculum = [iterations5x5, iterations6x6, iterations8x8, iterations16x16]
    iterationsDone = 0
    evaluation = []
    trainingInfoJson = {}
    # TODO load iterationsDone and decide where to continue training from
    print(len(curriculum))
    print(len(ENV_NAMES.ALL_ENVS))

    for i in range(len(curriculum)): # TODO fix this / update parameters and ALL_ENV etc
        iterationsDone = train.startTraining(iterationsDone + curriculum[i], args.model, ENV_NAMES.ALL_ENVS[i], args, txtLogger)
        evaluation.append(evaluate.evaluateAll(args.model, args))
        txtLogger.info(f"---Finished curriculum {ENV_NAMES.ALL_ENVS[i]} \n")
    # save iterations, training Duration
    duration = time.time() - startTime
    trainingInfoJson["numFrames"] = iterationsDone
    trainingInfoJson["trainingDuration"] = duration
    trainingInfoJson["scoreAfterEachEnv"] = evaluation

    modelPath = os.getcwd() + os.sep + "storage" + os.sep + args.model  # use this ??
    logFilePath = modelPath + os.sep + "status.json"
    with open(logFilePath, 'w') as f:
        f.write(json.dumps(trainingInfoJson, indent=4))
    return trainingInfoJson


def evaluateModel(trainingInfoJson):
    meanRewards = []
    # evaluateModel(trainingJson["scoreAfterEachEnv"])
    envRewards = {}
    evaluation = trainingInfoJson["scoreAfterEachEnv"]
    print("evaluation=", evaluation)
    i = 0
    for e in evaluation:
        meanReward = 0
        envRewards[ENV_NAMES.ALL_ENVS[i]] = []
        for evalEnv in ENV_NAMES.ALL_ENVS:
            envReward = float(e[evalEnv]["meanRet"])
            meanReward += envReward
            envRewards[ENV_NAMES.ALL_ENVS[i]].append(envReward)
        meanRewards.append(meanReward)
        i += 1
    print(envRewards)
    print(meanRewards)

#7027
