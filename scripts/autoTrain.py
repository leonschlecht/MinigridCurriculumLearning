import csv
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
    raise Exception("Next Env ??")


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
    evaluationResult = evaluate.evaluateAll(model, args)
    for evalEnv in ENV_NAMES.ALL_ENVS:
        reward += float(evaluationResult[evalEnv]["meanRet"])
    return reward


def startTraining(frames, model, env) -> int:
    """Starts the training. Returns the amount of iterations trained"""
    return train.main(frames, model, env, args)


def trainEachCurriculum(allCurricula, HORIZON_LENGTH, ITERATIONS_PER_CURRICULUM, i, pretrainIterations, epoch,
                        selectedModel, indexOfLastChosenCurriculum, curriculumChosenConsecutivelyTimes) -> int:
    """
    Simulates a horizon and returns the rewards obtained after evaluation the state at the end of the horizon
    """
    # TODO caps variables should be global after being read
    nameOfCurriculumI = getModelName(selectedModel, i)  # Save TEST_e1 --> TEST_e1_curric0
    copyAgent(src=selectedModel, dest=nameOfCurriculumI)
    jOffset = 0
    if indexOfLastChosenCurriculum == i:
        jOffset = curriculumChosenConsecutivelyTimes
    iterationsDone = pretrainIterations
    for j in range(jOffset, len(allCurricula[i])):
        iterationsDone = startTraining(iterationsDone + ITERATIONS_PER_CURRICULUM, nameOfCurriculumI,
                                       allCurricula[i][j])
        txtLogger.warning(f"Offset {iterationsDone}")
        if j == jOffset:
            copyAgent(src=nameOfCurriculumI,
                      dest=getModelWithCandidatePrefix(nameOfCurriculumI))  # save TEST_e1_curric0 -> + _CANDIDATE
        txtLogger.info(f"Trained iteration j {j} (offset {jOffset}")

    return 1
    # return evaluateAgent(nameOfCurriculumI)


def initializeRewards(N):
    """
    Loads the rewards given the state from previous training, or empty initializes them if first time run
    This assumes that the curricula are numbered from 1 to N (as opposed to using their names or something similar)
    """
    rewards = {}
    for i in range(N):
        rewards[str(i)] = []
    return rewards


def trainEnv(allCurricula):
    # ITERATIONS_PER_ENV = args.procs * args.frames_per_proc * 25 + 1  # roughly 100k with * 25
    ITERATIONS_PER_ENV = 25000
    HORIZON_LENGTH = ITERATIONS_PER_ENV * len(allCurricula[0])
    modelPath = os.getcwd() + "\\storage\\" + args.model
    logFilePath = modelPath + "\\status.json"

    if os.path.exists(logFilePath):
        with open(logFilePath, 'r') as f:
            trainingInfoJson = json.loads(f.read())

        txtLogger.info(trainingInfoJson)
        previousFrames = trainingInfoJson["numFrames"]
        startEpoch = trainingInfoJson["epochsDone"]

        rewards = trainingInfoJson["rewards"]
        txtLogger.info(f"Restoring from {startEpoch}... ")
        # copyAgent(src=selectedModel, dest=args.model + "_e" + str(startEpoch + 1))  # eSTART -> eSTART+1 # TODO ??

    else:
        PRE_TRAIN_FRAMES = 25000  # TODO set to 100k
        startEpoch = 1
        rewards = initializeRewards(len(allCurricula))  # dict of {"env1": [list of rewards], "env2": [rewards], ...}

        selectedModel = args.model + "_e" + str(0)

        modelPath = os.getcwd() + "\\storage\\" + selectedModel
        txtLogger.info("Pretraining. . .")
        previousFrames = startTraining(PRE_TRAIN_FRAMES, selectedModel, ENV_NAMES.DOORKEY_5x5)
        trainingInfoJson = {"selectedEnvs": [],
                            "bestCurriculaIds": [],
                            "rewards": rewards,
                            "epochsDone": startEpoch,
                            "numFrames": previousFrames}
        with open(logFilePath, 'w') as f:
            f.write(json.dumps(trainingInfoJson, indent=4))
        copyAgent(src=selectedModel, dest=args.model + "_e" + str(startEpoch))  # e0 -> e1; subsequent iterations do at the end of each epoch iteration

    print("Loading from ", modelPath)

    lastChosenCurriculum = -1
    indexOfLastChosenCurriculum = -1
    curriculumChosenConsecutivelyTimes = 0
    jOffset = 0

    for epoch in range(startEpoch, 3):
        selectedModel = args.model + "_e" + str(epoch)
        previousFrames += ITERATIONS_PER_ENV
        for i in range(len(allCurricula)):
            reward = trainEachCurriculum(allCurricula, HORIZON_LENGTH, ITERATIONS_PER_ENV, i, previousFrames, epoch,
                                         selectedModel, indexOfLastChosenCurriculum, curriculumChosenConsecutivelyTimes)
            rewards[str(i)].append(reward)
        indexOfLastChosenCurriculum = int(
            np.argmax([lst[-1] for lst in rewards.values()]))  # only access the latest reward
        txtLogger.info(f"Best results in epoch {epoch} came from curriculum {str(indexOfLastChosenCurriculum)}")

        copyAgent(src=getModelWithCandidatePrefix(getModelName(selectedModel, indexOfLastChosenCurriculum)),
                  dest=args.model + "_e" + str(epoch + 1))  # -> should be _e2, as it is the base for next iteration

        if epoch > 1 and indexOfLastChosenCurriculum == lastChosenCurriculum:
            curriculumChosenConsecutivelyTimes += 1
            if curriculumChosenConsecutivelyTimes > len(allCurricula[indexOfLastChosenCurriculum]):
                curriculumChosenConsecutivelyTimes = 0
        else:
            curriculumChosenConsecutivelyTimes = 0
        lastChosenCurriculum = indexOfLastChosenCurriculum

        trainingInfoJson["epochsDone"] = epoch + 1
        trainingInfoJson["numFrames"] = previousFrames
        trainingInfoJson["selectedEnvs"].append(
            allCurricula[indexOfLastChosenCurriculum][jOffset])  # TODO Test if this works properly
        trainingInfoJson["bestCurriculaIds"].append(indexOfLastChosenCurriculum)
        trainingInfoJson["rewards"] = rewards

        txtLogger.info(f"EPOCH: {epoch} SUCCESS")
        with open(logFilePath, 'w') as f:
            f.write(json.dumps(trainingInfoJson, indent=4))
        if epoch >= 2:
            break
    txtLogger.info("----TRAINING END-----")
    txtLogger.info(f"Best Curricula {trainingInfoJson['bestCurriculaIds']}")
    txtLogger.info(f"Trained in Envs {trainingInfoJson['selectedEnvs']}")
    txtLogger.info("Rewards:", rewards)
    txtLogger.info("-------------------")


def getModelName(model, curriculumNr):
    fullModelName = model
    fullModelName += "_curric" + str(curriculumNr)
    return fullModelName


def getModelWithCandidatePrefix(model):
    return model + "_CANDIDATE"


def copyAgent(src, dest):
    pathPrefix = os.getcwd() + '\\storage\\'
    fullSrcPath = pathPrefix + src
    fullDestPath = pathPrefix + dest
    if os.path.isdir(fullDestPath):
        txtLogger.warning(f"Path already exists! {fullDestPath} --> ???")
        # deleteDirectory(dest)
        # raise Exception(f"Path exists at {fullDestPath}! Copying agent failed")
    else:
        shutil.copytree(fullSrcPath, fullDestPath)
        txtLogger.info(f'Copied Agent! {src} ---> {dest}')


def deleteDirectory(directory):
    shutil.rmtree(os.getcwd() + "\\storage\\" + directory)


if __name__ == "__main__":
    args = parser.parse_args()
    args.mem = args.recurrence > 1

    txtLogger = utils.get_txt_logger(utils.get_model_dir(args.model))

    txtLogger.info(f"Device: {device}")

    uniformCurriculum = [ENV_NAMES.DOORKEY_5x5, ENV_NAMES.DOORKEY_6x6]
    other = [ENV_NAMES.DOORKEY_5x5, ENV_NAMES.DOORKEY_6x6, ENV_NAMES.DOORKEY_8x8]

    curricula = [uniformCurriculum]
    trainEnv(curricula)

"""
        txtLogger.info("Hello")
        # the folder will already exist with an empty log.txt file
        csv_path = os.path.join(os.getcwd() + "\\storage\\" + args.model, "log.csv")
        csv_file = open(csv_path, "a")
        csv_file, csv_logger = csv_file, csv.writer(csv_file)
        csv_logger.writerow(["epochsDone", "numFrames", "selectedEnvs", "bestCurriculaIds", "rewards"])
        csv_logger.writerow([0, PRE_TRAIN_FRAMES, [1,2,3,5,5,5,6], [], {"0": [1,2,3], "1": [4,5,6]}])
        csv_file.flush()
"""

"""
    # finish the environments that were skipped due to the ith curriculum being chosen jOffset times


    additionalOffset = len(allCurricula[i]) - jOffset
    for j in range(jOffset):
        startTraining(epoch * HORIZON_LENGTH + FRAMES_PER_CURRICULUM * (additionalOffset + j + 1)
                      + previousFrames, nameOfCurriculumI, allCurricula[i][j])
        txtLogger.info(f"__Trained iteration j {j} offset {jOffset}")
        txtLogger.warning(
            f"Offset {epoch * HORIZON_LENGTH + FRAMES_PER_CURRICULUM * (additionalOffset + j + 1) + previousFrames}")
"""
