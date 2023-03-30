import os
import argparse
import json
import numpy as np
import shutil
import utils
from scripts import evaluate, train
from constants import ENV_NAMES
from utils import device

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument("--algo", default="ppo",
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--env",
                    help="environemnt ")  # TODO
parser.add_argument("--pretraining", default=False, action="store_true", help="use pretraining on new model")
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
parser.add_argument("--frames", type=int, default=10 ** 7,
                    help="number of frames of training (default: 1e7)")

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


def evaluateAgent(model):
    reward = 0
    evaluationResult = evaluate.evaluateAll(model, args)
    for evalEnv in ENV_NAMES.ALL_ENVS:
        reward += float(evaluationResult[evalEnv]["meanRet"])  # TODO use weights, maybe depending on progress
        # TODO maybe use something else like Value / Policy Loss / Frames
    return reward


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


def startTraining(frames, model, env):
    train.main(frames, model, env, args)


def trainEachCurriculum(allCurricula, HORIZON_LENGTH, FRAMES_PER_CURRICULUM, i, previousFrames, jOffset, epoch,
                        currentModel, selectedModel):
    pass


def trainEnv(allCurricula):
    # ITERATIONS_PER_CURRICULUM = args.procs * args.frames_per_proc * 25 + 1  # roughly 100k with * 25
    ITERATIONS_PER_CURRICULUM = 20000
    HORIZON_LENGTH = ITERATIONS_PER_CURRICULUM * len(allCurricula[0])  # TODO maybe move this inside the loop
    rewards = {}
    for i in range(len(allCurricula)):
        rewards[str(i)] = []
    selectedModel = args.model
    bestCurricula = []
    modelPath = os.getcwd() + "\\storage\\" + selectedModel
    txtLogger.info("Pretraining. . .")
    PRE_TRAIN_FRAMES = 25000
    startTraining(PRE_TRAIN_FRAMES, selectedModel, ENV_NAMES.DOORKEY_5x5)
    status = utils.get_status(modelPath)
    previousFrames = status["num_frames"]

    FRAMES_PER_CURRICULUM = ITERATIONS_PER_CURRICULUM
    envHistory = []  # TODO Load from file
    lastChosenCurriculum = -1
    indexOfLastChosenCurriculum = -1
    curriculumChosenConsecutivelyTimes = 0
    jOffset = 0
    for epoch in range(6):
        for i in range(len(allCurricula)):
            currentModel = selectedModel + '_e' + str(epoch) + '_i' + str(i)  # TEST --> TEST_e0_i0
            copyAgent(selectedModel, currentModel)
            jOffset = 0
            if indexOfLastChosenCurriculum == i:
                jOffset = curriculumChosenConsecutivelyTimes
            txtLogger.info(f"J offset {jOffset}")
            for j in range(jOffset, len(allCurricula[i])):
                startTraining(epoch * HORIZON_LENGTH + FRAMES_PER_CURRICULUM * (j + 1 - jOffset) + previousFrames,
                              currentModel, allCurricula[i][j])
                txtLogger.warn(
                    f"Offset {epoch * HORIZON_LENGTH + FRAMES_PER_CURRICULUM * (j + 1 - jOffset) + previousFrames}")
                if j == jOffset:
                    copyAgent(src=currentModel, dest=selectedModel + "_CANDIDATE_e" + str(epoch) + "_" + str(
                        i))  # TEST_e0_i0 -> TEST_CANDIDATE_e0_i0
                txtLogger.info(f"Trained iteration j {j} (offset {jOffset}")

            # finish the environments that were skipped due to the ith curriculum being chosen jOffset times
            additionalOffset = len(allCurricula[i]) - jOffset
            for j in range(jOffset):
                startTraining(epoch * HORIZON_LENGTH + FRAMES_PER_CURRICULUM * (additionalOffset + j + 1)
                              + previousFrames, currentModel, allCurricula[i][j])
                txtLogger.info(f"__Trained iteration j {j} offset {jOffset}")
                txtLogger.warn(
                    f"Offset {epoch * HORIZON_LENGTH + FRAMES_PER_CURRICULUM * (additionalOffset + j + 1) + previousFrames}")

            rewards[str(i)].append(evaluateAgent(currentModel))
        indexOfLastChosenCurriculum = np.argmax(rewards)
        txtLogger.info(f"Best results in epoch {epoch} came from curriculum {str(indexOfLastChosenCurriculum)}")
        bestCurricula.append(indexOfLastChosenCurriculum)
        # Take snapshot and make it main model
        copyAgent(selectedModel + "_CANDIDATE_" + str(indexOfLastChosenCurriculum),
                  selectedModel)  # TEST_CANDIDATE_e0_i0 --> TEST
        envHistory.append(allCurricula[indexOfLastChosenCurriculum][jOffset])
        if epoch > 0 and indexOfLastChosenCurriculum == lastChosenCurriculum:
            curriculumChosenConsecutivelyTimes += 1
            if curriculumChosenConsecutivelyTimes > len(allCurricula[indexOfLastChosenCurriculum]):
                curriculumChosenConsecutivelyTimes = 0
        else:
            curriculumChosenConsecutivelyTimes = 0
        lastChosenCurriculum = indexOfLastChosenCurriculum

        txtLogger.info(f"EPOCH: {epoch} SUCCESS")
    txtLogger.info("----TRAINING END-----")
    txtLogger.info(f"Best Curricula {bestCurricula}")
    txtLogger.info("Trained in Envs:", envHistory)
    txtLogger.info("Rewards:", rewards)
    txtLogger.info("-------------------")


def copyAgent(src, dest):
    pathPrefix = os.getcwd() + '\\storage\\'
    fullSrcPath = pathPrefix + src
    fullDestPath = pathPrefix + dest

    if os.path.isdir(fullDestPath):
        # txtLogger.warn(f"Path already exists! {fullDestPath}")
        raise Exception(f"Path exists at {fullDestPath}! Copying agent failed")
    else:
        shutil.copytree(fullSrcPath, fullDestPath)
        print('Copied Agent! ' + src + ' to ' + dest)


def deleteDirectory(directory):
    shutil.rmtree(os.getcwd() + "\\storage\\" + directory)


if __name__ == "__main__":
    args = parser.parse_args()
    args.mem = args.recurrence > 1

    txtLogger = utils.get_txt_logger(utils.get_model_dir(args.model))

    print(f"Device: {device}")

    uniformCurriculum = [ENV_NAMES.DOORKEY_5x5, ENV_NAMES.DOORKEY_6x6, ENV_NAMES.DOORKEY_5x5]
    other = [ENV_NAMES.DOORKEY_5x5, ENV_NAMES.DOORKEY_6x6, ENV_NAMES.DOORKEY_8x8]
    # TODO Add Curricula: Adaptive, Random Order
    curricula = [uniformCurriculum, other]
    trainEnv(curricula)
