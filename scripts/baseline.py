import utils
from scripts import train, evaluate
from utils import initializeParser, ENV_NAMES
import time
import os
import json

"""
    Trains each curriculum in order for a pre-set amount of iterations
"""


def doBaselineTraining():
    TOTAL_ITERATIONS = 20000000
    iterations5x5 = 50000
    iterations6x6 = 500000
    iterations8x8 = 4000000
    iterations16x16 = TOTAL_ITERATIONS - iterations8x8 - iterations6x6 - iterations5x5
    curriculum = [iterations5x5, iterations6x6, iterations8x8, iterations16x16]
    iterationsDone = 0
    txtLogger = utils.get_txt_logger(utils.get_model_dir(args.model))
    evaluation = []
    trainingInfoJson = {}

    for i in range(len(curriculum)):
        iterationsDone = train.main(5000, args.model, ENV_NAMES.ALL_ENVS[i], args)
        evaluation[i] = evaluate.evaluateAll(args.model, args)
        txtLogger.info(f"---Finished curriculum {ENV_NAMES.ALL_ENVS[i]} \n")
    # save iterations, training Duration
    duration = time.time() - startTime
    trainingInfoJson["numFrames"] = iterationsDone
    trainingInfoJson["trainingDuration"] = duration
    trainingInfoJson["scoreAfterEachEnv"] = evaluation

    modelPath = os.getcwd() + "\\storage\\" + args.model  # use this ??
    logFilePath = modelPath + "\\status.json"
    with open(logFilePath, 'w') as f:
        f.write(json.dumps(trainingInfoJson, indent=4))
    return trainingInfoJson


def evaluateModel(trainingInfoJson):
    meanRewards = []
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


if __name__ == "__main__":
    parser = initializeParser()
    args = parser.parse_args()
    args.mem = args.recurrence > 1
    startTime = time.time()
    trainingJson = doBaselineTraining()#7027

    evaluateModel(trainingJson["scoreAfterEachEnv"])
