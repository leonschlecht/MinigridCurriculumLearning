import os
import argparse
import json
import numpy as np
import shutil
import utils
from scripts import evaluate, train
from constants import ENV_NAMES
from utils import device
import time

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument("--algo", default="ppo",
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=2,
                    help="number of updates between two saves (default: 2, 0 means no saving)")
parser.add_argument("--procs", type=int, default=32,
                    help="number of processes (default: 32)")

# Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--frames-per-proc", type=int, default=None,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate (default: 0.001)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")

# Evaluation Arguments
parser.add_argument("--episodes", type=int, default=10,
                    help="number of episodes of evaluation (default: 10)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--worst-episodes-to-show", type=int, default=10,
                    help="how many worst episodes to show")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")


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


def evaluateAgent(model) -> int:
    """
    Evaluates and calculates the average performance in ALL environments
    :param model: the name of the model
    :return: the average reward
    """
    reward = 0
    evaluationResult = evaluate.evaluateAll(model, args) # TODO decide if argmax or not
    for evalEnv in ENV_NAMES.ALL_ENVS:
        reward += float(evaluationResult[evalEnv]["meanRet"])
    return reward


def startTraining(frames, model, env) -> int:
    """
    Starts the training.
    :param frames:
    :param model:
    :param env:
    :return: Returns the amount of iterations trained
    """
    return train.main(frames, model, env, args)


def trainEachCurriculum(allCurricula, i, iterationsDone, selectedModel, jOffset) -> int:
    """
    Simulates a horizon and returns the rewards obtained after evaluating the state at the end of the horizon

    :param jOffset:
    :param allCurricula:
    :param i:
    :param iterationsDone:
    :param selectedModel:
    :return:
    """
    nameOfCurriculumI = getModelName(selectedModel, i)  # Save TEST_e1 --> TEST_e1_curric0
    copyAgent(src=selectedModel, dest=nameOfCurriculumI)
    for j in range(jOffset, len(allCurricula[i])):
        iterationsDone = startTraining(iterationsDone + ITERATIONS_PER_ENV, nameOfCurriculumI,
                                       allCurricula[i][j])
        txtLogger.info(f"Iterations Done {iterationsDone}")
        if j == jOffset:
            copyAgent(src=nameOfCurriculumI,
                      dest=getModelWithCandidatePrefix(nameOfCurriculumI))  # save TEST_e1_curric0 -> + _CANDIDATE
        txtLogger.info(f"Trained iteration j {j} (offset {jOffset}) of curriculum {i}")

    return evaluateAgent(nameOfCurriculumI)


def initializeRewards(N):
    """
    Loads the rewards given the state from previous training, or empty initializes them if first time run
    This assumes that the curricula are numbered from 1 to N (as opposed to using their names or something similar)
    """
    rewards = {}
    for i in range(N):
        rewards[str(i)] = []
    return rewards


def getCurriculaEnvDetails(allCurricula) -> dict:
    full_env_list = {}
    for i in range(len(allCurricula)):
        full_env_list[i] = allCurricula[i]
    return full_env_list


def updateTrainingInfo(trainingInfoJson, logFilePath, epoch, iterationsDoneSoFar, selectedEnv, currentBestCurriculum,
                       rewards, curriculaEnvDetails) -> None:
    """

    :param trainingInfoJson:
    :param logFilePath:
    :param epoch:
    :param iterationsDoneSoFar:
    :param selectedEnv:
    :param currentBestCurriculum:
    :param rewards:
    :param curriculaEnvDetails:
    """
    trainingInfoJson["epochsDone"] = epoch + 1
    trainingInfoJson["numFrames"] = iterationsDoneSoFar
    trainingInfoJson["selectedEnvs"].append(selectedEnv)  # TODO Test if this works properly
    trainingInfoJson["bestCurriculaIds"].append(currentBestCurriculum)
    trainingInfoJson["rewards"] = rewards
    trainingInfoJson["curriculaEnvDetails"]["epoch" + str(epoch)] = curriculaEnvDetails

    with open(logFilePath, 'w') as f:
        f.write(json.dumps(trainingInfoJson, indent=4))

    txtLogger.info(f"Best results in epoch {epoch} came from curriculum {currentBestCurriculum}")
    txtLogger.info(f"CurriculaEnvDetails {trainingInfoJson['curriculaEnvDetails']}")
    txtLogger.info(f"\nEPOCH: {epoch} SUCCESS\n")


def calculateConsecutivelyChosen(consecutiveCount, currentBestCurriculum, lastChosenCurriculum, curriculaList) -> int:
    if consecutiveCount + 1 >= len(curriculaList[0]) or currentBestCurriculum != lastChosenCurriculum:
        return 0
    return consecutiveCount + 1


def printFinalLogs(trainingInfoJson, startingTime) -> None:
    """

    :param trainingInfoJson:
    """
    txtLogger.info("----TRAINING END-----")
    txtLogger.info(f"Best Curricula {trainingInfoJson['bestCurriculaIds']}")
    txtLogger.info(f"Trained in Envs {trainingInfoJson['selectedEnvs']}")
    txtLogger.info(f"Rewards: {trainingInfoJson['rewards']}")
    txtLogger.info(f"Time ended at {time.time()} , total training time: {-startingTime + time.time()}")
    txtLogger.info("-------------------\n\n")


def initTrainingInfo(logFilePath, rewards, pretrainingIterations):
    trainingInfo = {"selectedEnvs": [],
                    "bestCurriculaIds": [],
                    "curriculaEnvDetails": {},
                    "rewards": rewards,
                    "epochsDone": 1,
                    "numFrames": pretrainingIterations}
    with open(logFilePath, 'w') as f:
        f.write(json.dumps(trainingInfo, indent=4))
    return trainingInfo


def calculateJOffset(curriculumChosenConsecutivelyTimes, isZero) -> int:
    if isZero:
        return 0
    return curriculumChosenConsecutivelyTimes


def startCurriculumTraining(allCurricula: list) -> None:
    """

    :param allCurricula:
    """
    assert len(allCurricula) > 0

    trainStart = time.time()
    modelPath = os.getcwd() + "\\storage\\" + args.model  # use this ??
    logFilePath = modelPath + "\\status.json"

    if os.path.exists(logFilePath):
        with open(logFilePath, 'r') as f:
            trainingInfoJson = json.loads(f.read())

        iterationsDoneSoFar = trainingInfoJson["numFrames"]
        startEpoch = trainingInfoJson["epochsDone"]
        rewards = trainingInfoJson["rewards"]

        txtLogger.info(f"Continung training from epoch {startEpoch}... ")
    else:
        startEpoch = 1
        rewards = initializeRewards(len(allCurricula))  # dict of {"env1": [list of rewards], "env2": [rewards], ...}
        selectedModel = args.model + "\\epoch_" + str(0)
        txtLogger.info("Pretraining. . .")
        iterationsDoneSoFar = startTraining(PRE_TRAIN_FRAMES, selectedModel, ENV_NAMES.DOORKEY_5x5)
        trainingInfoJson = initTrainingInfo(logFilePath, rewards, iterationsDoneSoFar)
        copyAgent(src=selectedModel, dest=args.model + "\\epoch_" + str(
            startEpoch))  # e0 -> e1; subsequent iterations do at the end of each epoch iteration

    lastChosenCurriculum = None
    curriculumChosenConsecutivelyTimes = 0

    for epoch in range(startEpoch, 11):
        selectedModel = args.model + "\\epoch_" + str(epoch)
        for i in range(len(allCurricula)):
            jOffset = calculateJOffset(curriculumChosenConsecutivelyTimes, i == lastChosenCurriculum)
            reward = trainEachCurriculum(allCurricula, i, iterationsDoneSoFar, selectedModel, jOffset)
            rewards[str(i)].append(reward)
        iterationsDoneSoFar += ITERATIONS_PER_ENV
        currentBestCurriculum = int(np.argmax([lst[-1] for lst in rewards.values()]))  # only access the latest reward

        copyAgent(src=getModelWithCandidatePrefix(getModelName(selectedModel, currentBestCurriculum)),
                  dest=args.model + "\\epoch_" + str(epoch + 1))  # the model for the next epoch

        curriculumChosenConsecutivelyTimes = calculateConsecutivelyChosen(curriculumChosenConsecutivelyTimes,
                                                                          currentBestCurriculum, lastChosenCurriculum,
                                                                          allCurricula)
        lastChosenCurriculum = currentBestCurriculum

        updateTrainingInfo(trainingInfoJson, logFilePath, epoch, iterationsDoneSoFar,
                           allCurricula[currentBestCurriculum][jOffset], currentBestCurriculum, rewards,
                           getCurriculaEnvDetails(allCurricula))

    printFinalLogs(trainingInfoJson, trainStart)


def getModelName(model, curriculumNr) -> str:
    return model + "_curric" + str(curriculumNr)


def getModelWithCandidatePrefix(model) -> str:
    return model + "_CANDIDATE"


def copyAgent(src, dest) -> None:
    pathPrefix = os.getcwd() + '\\storage\\'
    fullSrcPath = pathPrefix + src
    fullDestPath = pathPrefix + dest
    if os.path.isdir(fullDestPath):
        raise Exception(f"Path exists at {fullDestPath}! Copying agent failed")
    else:
        shutil.copytree(fullSrcPath, fullDestPath)
        txtLogger.info(f'Copied Agent! {src} ---> {dest}')


def deleteModel(directory) -> None:
    """
    :param directory: of the model to be deleted
    """
    shutil.rmtree(os.getcwd() + "\\storage\\" + directory)


def adaptiveCurriculum():
    iterationsDoneSoFar = 0  # TODO load
    currentEnv = ENV_NAMES.DOORKEY_6x6
    nextEnvi = nextEnv(currentEnv)
    for epoch in range(1, 25):
        iterationsDoneSoFar = startTraining(iterationsDoneSoFar + ITERATIONS_PER_ENV, args.model, currentEnv)
        iterationsDoneSoFar = startTraining(iterationsDoneSoFar + ITERATIONS_PER_ENV, args.model, nextEnvi)
        iterationsDoneSoFar = startTraining(iterationsDoneSoFar + ITERATIONS_PER_ENV, args.model, currentEnv)

        score = evaluateAgent(args.model)
        if score <= 1:
            currentEnv = ENV_NAMES.DOORKEY_5x5
        elif score <= 2:
            currentEnv = ENV_NAMES.DOORKEY_6x6
        elif score <= 3:
            currentEnv = ENV_NAMES.DOORKEY_8x8
        else:
            currentEnv = ENV_NAMES.DOORKEY_16x16
        nextEnvi = nextEnv(currentEnv)
        txtLogger.info(f"score in ep {epoch}: {score} ---> next Env: {currentEnv}; iterations: {iterationsDoneSoFar}")


    print("Done")


if __name__ == "__main__":
    args = parser.parse_args()
    args.mem = args.recurrence > 1

    txtLogger = utils.get_txt_logger(utils.get_model_dir(args.model))
    txtLogger.info(f"Device: {device}")

    uniformCurriculum = [ENV_NAMES.DOORKEY_5x5, ENV_NAMES.DOORKEY_6x6, ENV_NAMES.DOORKEY_8x8, ENV_NAMES.DOORKEY_16x16]
    focus8 = [ENV_NAMES.DOORKEY_8x8, ENV_NAMES.DOORKEY_8x8, ENV_NAMES.DOORKEY_8x8, ENV_NAMES.DOORKEY_6x6]
    mix16_8 = [ENV_NAMES.DOORKEY_16x16, ENV_NAMES.DOORKEY_16x16, ENV_NAMES.DOORKEY_8x8, ENV_NAMES.DOORKEY_8x8]
    idk = [ENV_NAMES.DOORKEY_16x16, ENV_NAMES.DOORKEY_8x8, ENV_NAMES.DOORKEY_16x16, ENV_NAMES.DOORKEY_6x6]

    curricula = [uniformCurriculum, focus8, mix16_8, idk]

    ITERATIONS_PER_ENV = 150000
    PRE_TRAIN_FRAMES = 100000
    HORIZON_LENGTH = ITERATIONS_PER_ENV * len(curricula[0])

    # startCurriculumTraining(curricula)
    adaptiveCurriculum()
