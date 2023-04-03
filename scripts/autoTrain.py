import os
import argparse
import json
import numpy as np
import utils
from scripts import evaluate, train
from utils import ENV_NAMES
from utils import device, getModelWithCandidatePrefix
import time


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
    evaluationResult = evaluate.evaluateAll(model, args)  # TODO decide if argmax or not
    for evalEnv in ENV_NAMES.ALL_ENVS:
        reward += float(evaluationResult[evalEnv]["meanRet"])
    return reward


def startTraining(frames, model, env) -> int:
    """
    Starts the training.
    :return: Returns the amount of iterations trained
    """
    return train.main(frames, model, env, args)


def trainEachCurriculum(allCurricula, i, iterationsDone, selectedModel, jOffset) -> int:
    """
    Simulates a horizon and returns the rewards obtained after evaluating the state at the end of the horizon
    """
    nameOfCurriculumI = utils.getModelName(selectedModel, i)  # Save TEST_e1 --> TEST_e1_curric0
    utils.copyAgent(src=selectedModel, dest=nameOfCurriculumI)
    for j in range(jOffset, len(allCurricula[i])):
        iterationsDone = startTraining(iterationsDone + ITERATIONS_PER_ENV, nameOfCurriculumI,
                                       allCurricula[i][j])
        txtLogger.info(f"Iterations Done {iterationsDone}")
        if j == jOffset:
            utils.copyAgent(src=nameOfCurriculumI,
                            dest=utils.getModelWithCandidatePrefix(
                                nameOfCurriculumI))  # save TEST_e1_curric0 -> + _CANDIDATE
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
    trainingInfoJson["epochsDone"] = epoch + 1
    trainingInfoJson["numFrames"] = iterationsDoneSoFar
    trainingInfoJson["selectedEnvs"].append(
        selectedEnv)  # TODO Test if this works properly / also maybe remove bcs redundant
    trainingInfoJson["bestCurriculaIds"].append(currentBestCurriculum)
    trainingInfoJson["rewards"] = rewards
    currentScore = evaluateAgent(args.model + "\\epoch_" + str(epoch + 1))
    trainingInfoJson["actualPerformance"].append([currentScore, selectedEnv])
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
                    "actualPerformance": [],
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
    Starts The RH Curriculum Training
    :param allCurricula: the curriculums for which the training will be done
    """
    assert len(allCurricula) > 0

    trainStart = time.time()
    modelPath = os.getcwd() + "\\storage\\" + args.model
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
        utils.copyAgent(src=selectedModel, dest=args.model + "\\epoch_" + str(
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

        utils.copyAgent(src=getModelWithCandidatePrefix(utils.getModelName(selectedModel, currentBestCurriculum)),
                        dest=args.model + "\\epoch_" + str(epoch + 1))  # the model for the next epoch

        curriculumChosenConsecutivelyTimes = calculateConsecutivelyChosen(curriculumChosenConsecutivelyTimes,
                                                                          currentBestCurriculum, lastChosenCurriculum,
                                                                          allCurricula)
        lastChosenCurriculum = currentBestCurriculum

        updateTrainingInfo(trainingInfoJson, logFilePath, epoch, iterationsDoneSoFar,
                           allCurricula[currentBestCurriculum][jOffset], currentBestCurriculum, rewards,
                           getCurriculaEnvDetails(allCurricula))
        # TODO : EVOLUTIONARY
        # Muss man speichern, was vorher gut war?
        # Muss man die bestimmten Envs schwierigkeitsgrade zuweisen, um die neuen evol. Curricula zu generieren?
        #
        # Man kann evtl. speichern, wann das Model den besten Reward erhalten hat, um darauf zurückzugriefen; aber overfitting ist ja auch so eine sache hier weil man keine test/train untescheidung hat
        # wir haben die rewards für die jeweiligen curricula;
        # Ist der interessante Teil das gute erstellen von curricula mithilfe von EA; oder einfach nur dass man
        # eine gute Gesamtidee findet?

    printFinalLogs(trainingInfoJson, trainStart)


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


def adaptiveCurriculum():
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
            iterationsDoneSoFar = startTraining(iterationsDoneSoFar + ITERATIONS_PER_ENV, args.model, env)

        evaluationScore = evaluateAgent(args.model)
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


def evaluateCurriculumResults():
    pass
    # ["actualPerformance"][0] ---> zeigt den avg reward des models zu jedem übernommenen Snapshot
    # ["actualPerformance"][1] ---> zeigt die zuletzt benutzte Umgebung zu dem Zeitpunkt an
    #
    tmp = []
    i = 0
    for reward, env in tmp:
        print(reward, env)
        i += 1

    # Dann wollen wir sehen, wie das curriculum zu dem jeweiligen zeitpunkt ausgesehen hat.
    # # Aber warum? Und wie will man das nach 20+ durhcläufen plotten


if __name__ == "__main__":
    args = utils.initializeParser()
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

    startCurriculumTraining(curricula)
    # adaptiveCurriculum()

    # evaluateCurriculumResults(trainingInfoJson)
