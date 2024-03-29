import os

import numpy as np

from curricula import train, evaluate, RollingHorizon
from utils import getEnvFromDifficulty, storage
from utils.curriculumHelper import *


class allParalell:
    """
    This class contains multiple baseline variants. They train for X iterations and then evaluate and save the results / logs
    - SPCL: add --trainAllParalell --allSimultaneous
    - AllParallel with all envs simultaneously --trainAllParalell
    - AllParallel as a repeating curriculum --trainAllParalell --asCurric
    - PPO Only trying to solve an env with regular evaluations add --trainAllParalell --ppoEnv (envIndex)
    """
    def __init__(self, txtLogger, startTime, cmdLineString: str, args, modelName):
        self.difficultyStepSize = args.difficultyStepsize
        self.args = args
        self.cmdLineString = cmdLineString
        self.lastEpochStartTime = startTime
        self.envDifficulty: float = 1.0
        self.seed = args.seed
        self.constMaxsteps = args.constMaxsteps
        if args.dynamicObstacle:
            self.allEnvs = ENV_NAMES.DYNAMIC_OBST_ENVS
        else:
            self.allEnvs = ENV_NAMES.DOORKEY_ENVS
        self.paraEnvs = len(self.allEnvs)

        self.ITERATIONS_PER_EVALUATE = args.iterPerEnv
        self.iterationsDone = 0
        self.txtLogger = txtLogger
        TOTAL_ITERATIONS = args.trainingIterations * 2
        self.totalEpochs = TOTAL_ITERATIONS // self.ITERATIONS_PER_EVALUATE
        self.trainingTime = 0
        self.model = modelName
        self.isSPLCL = not args.allSimultaneous
        self.asCurriculum = args.asCurriculum
        self.ppoEnv = args.ppoEnv
        self.ppoSingleEnv = self.ppoEnv != -1
        if self.ppoSingleEnv:
            assert 0 <= self.ppoEnv < len(self.allEnvs)

        self.selectedModel = self.model + os.sep + "model"

        self.stepMaxReward = calculateCurricStepMaxReward(self.allEnvs, args.noRewardShaping)

        self.trainingInfoJson = {}
        self.logFilePath = storage.getLogFilePath(["storage", self.model, "status.json"])

        self.curriculaEnvDetails = {}
        self.modelExists = os.path.exists(self.logFilePath)
        self.latestReward = 0
        self.startEpoch = 1

        if not self.isSPLCL:
            print("AllPara normal or NoCurric or ppo only")
            if self.ppoSingleEnv:
                self.initialEnvNames = self.updateEnvNamesNoAdjustment(self.envDifficulty, self.allEnvs, self.ppoEnv)
            else:
                self.initialEnvNames = self.updateEnvNamesNoAdjustment(self.envDifficulty, self.allEnvs, 0)
        else:
            print("SPLCL")
            self.initialEnvNames = self.initializeEnvNames(self.allEnvs, self.envDifficulty)
        if self.modelExists:
            self.loadTrainingInfo()
        else:
            self.trainingInfoJson = RollingHorizon.initTrainingInfo(self.cmdLineString, self.logFilePath, self.seed,
                                                                    self.stepMaxReward, None, self.allEnvs, self.args)

        self.trainEachCurriculum(self.startEpoch, self.totalEpochs, self.iterationsDone, self.initialEnvNames)

    def trainEachCurriculum(self, startEpoch: int, totalEpochs: int, iterationsDone: int, initialEnvNames: list):
        envNames = initialEnvNames
        print("training will go on until", totalEpochs)
        currentStep = 1  # helper param to determine at what point in an epoch we are (used for AllParallel as Curric)
        for epoch in range(startEpoch, totalEpochs):
            iterationsDone = train.startTraining(iterationsDone + self.ITERATIONS_PER_EVALUATE, iterationsDone,
                                                 self.selectedModel, envNames, self.args, self.txtLogger)
            if epoch == 0:
                self.ITERATIONS_PER_EVALUATE = iterationsDone
                self.txtLogger.info(f"Exact iterations set: {iterationsDone} ")
            reward = (evaluate.evaluateAgent(self.selectedModel, self.envDifficulty, self.args, self.txtLogger, self.allEnvs))
            self.trainingInfoJson[rawRewardsKey][f"epoch_{epoch}"] = reward
            reward = np.sum(reward)
            if not self.constMaxsteps:
                self.envDifficulty = calculateEnvDifficulty(iterationsDone, self.difficultyStepSize, self.allEnvs)
            oldEnvNames = envNames.copy()  # used for the logs
            if not self.isSPLCL:
                if self.asCurriculum:
                    envNames = self.updateEnvNamesNoAdjustment(self.envDifficulty, self.allEnvs, currentStep)
                else:
                    if self.ppoSingleEnv:
                        envNames = self.updateEnvNamesNoAdjustment(self.envDifficulty, self.allEnvs, self.ppoEnv)
                    else:
                        envNames = self.updateEnvNamesNoAdjustment(self.envDifficulty, self.allEnvs)
                currentStep += 1
                if currentStep >= len(self.allEnvs):
                    currentStep = 0
            else:
                envNames = self.updateEnvNamesDynamically(envNames, self.envDifficulty, self.seed + epoch, reward)
            self.updateTrainingInfo(self.trainingInfoJson, epoch, oldEnvNames, reward, self.envDifficulty, iterationsDone)
            self.logInfoAfterEpoch(epoch, reward, self.txtLogger, totalEpochs)
            self.txtLogger.info(f"reward {reward}")

    @staticmethod
    def logInfoAfterEpoch(epoch, reward, txtLogger, totalEpochs):
        """
        Logs relevant training info after a training epoch is done and the trainingInfo was updated
        :param reward:
        :param txtLogger: the reference to the logger which logs all the info happening during training
        :param totalEpochs: the amount of epochs the training will go on for
        :param epoch: the current epoch nr
        :return:
        """
        currentEpoch = "epoch_" + str(epoch)
        txtLogger.info(f"Current rewards after {currentEpoch}: {reward}")
        txtLogger.info(f"\nEPOCH: {epoch} SUCCESS (total: {totalEpochs})\n ")

    @staticmethod
    def updateEnvNamesNoAdjustment(difficulty, envList: list, currentStep=-1) -> list:
        envNames = []
        for j in range(len(envList)):
            if currentStep != -1:
                index = currentStep
            else:
                index = j
            envNames.append(getEnvFromDifficulty(index, envList, difficulty))
        return envNames

    @staticmethod
    def initializeEnvNames(envList: list, startDifficulty=1.0) -> list:
        """
        Initialize an env list for the training with the easiest index
        :param envList:
        :param startDifficulty:
        :return:
        """
        envNames = []
        for j in range(len(envList)):
            envNames.append(getEnvFromDifficulty(0, envList, startDifficulty))
        return envNames

    @staticmethod
    def getHarderEnv(envList, envName) -> str:
        index = envList.index(envName) + 1
        if index >= len(envList):
            index = len(envList) - 1
        return envList[index]

    @staticmethod
    def getEasierEnv(envList, envName) -> str:
        index = envList.index(envName) - 1
        if index < 0:
            index = 0
        return envList[index]

    def updateEnvNamesDynamically(self, currentEnvNames: list, newDifficulty: float, seed: int, reward: float) -> list:
        """
        SPLCL Update method depending on the progress decide which env to use
        :param currentEnvNames:
        :param newDifficulty:
        :param seed:
        :param reward:
        :return:
        """
        envNames = currentEnvNames
        assert len(currentEnvNames) == len(self.allEnvs)
        np.random.seed(seed)
        randomIndexSample = np.random.choice(range(len(currentEnvNames)), size=self.paraEnvs, replace=False)
        print("SPLCLCL reward", reward)
        if reward > self.stepMaxReward * .85:
            nextStep = "goUp"
        elif reward > self.stepMaxReward * .5:
            nextStep = "stay"
        else:
            nextStep = "goDown"
        if nextStep == "stay":
            for i in range(len(envNames)):
                cutEnv = envNames[i].split("-custom")[0]
                envNames[i] = cutEnv + ENV_NAMES.CUSTOM_POSTFIX + str(newDifficulty)
        else:
            for i in randomIndexSample:
                cutEnv = envNames[i].split("-custom")[0]
                if nextStep == "goDown":
                    nextEnv = self.getEasierEnv(self.allEnvs, cutEnv)
                elif nextStep == "goUp":
                    nextEnv = self.getHarderEnv(self.allEnvs, cutEnv)
                else:
                    raise Exception("Invalid new difficulty")
                envNames[i] = nextEnv + ENV_NAMES.CUSTOM_POSTFIX + str(newDifficulty)
        for i in range(len(envNames)):
            cutEnv = envNames[i].split("-custom")[0]
            envNames[i] = cutEnv + ENV_NAMES.CUSTOM_POSTFIX + str(newDifficulty)
        return envNames

    def updateTrainingInfo(self, trainingInfoJson, epoch, envNames, reward, difficulty, framesDone):
        currentEpoch = "epoch_" + str(epoch)

        trainingInfoJson[epochsDone] = epoch + 1
        trainingInfoJson[numFrames] = framesDone
        trainingInfoJson[snapshotScoreKey].append(reward)
        trainingInfoJson[curriculaEnvDetailsKey][currentEpoch] = envNames
        trainingInfoJson[difficultyKey].append(difficulty)
        now = datetime.now()

        timeSinceLastEpoch = (now - self.lastEpochStartTime).total_seconds()
        trainingInfoJson[epochTrainingTime].append(timeSinceLastEpoch)
        trainingInfoJson[sumTrainingTime] += timeSinceLastEpoch
        trainingInfoJson["currentListOfCurricula"] = envNames

        saveTrainingInfoToFile(self.logFilePath, trainingInfoJson)
        self.lastEpochStartTime = now

    def loadTrainingInfo(self):
        with open(self.logFilePath, 'r') as f:
            self.trainingInfoJson = json.loads(f.read())

        if len(self.trainingInfoJson[difficultyKey]) > 0:
            self.envDifficulty = self.trainingInfoJson[difficultyKey][-1]
        self.iterationsDone = self.trainingInfoJson[numFrames]
        # self.initialEnvNames = self.trainingInfoJson["nextEnvList"]  # TODO test this
        self.startEpoch = self.trainingInfoJson[epochsDone]
        if len(self.trainingInfoJson[snapshotScoreKey]) > 0:
            self.latestReward = self.trainingInfoJson[snapshotScoreKey][-1]
        self.txtLogger.info(f'Reloading state\nContinuing from epoch {self.startEpoch}, seed: {self.seed}')
