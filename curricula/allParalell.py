import os

import utils
from curricula import train, evaluate, RollingHorizon
from utils import ENV_NAMES, getEnvFromDifficulty, storage
from utils.curriculumHelper import *


class allParalell:
    def __init__(self, txtLogger, startTime, cmdLineString: str, args):
        # random.seed(args.seed)
        self.args = args
        self.cmdLineString = cmdLineString
        self.lastEpochStartTime = startTime
        self.envDifficulty = 0
        self.seed = args.seed

        self.ITERATIONS_PER_EVALUATE = args.iterPerEnv
        self.iterationsDone = 0
        self.txtLogger = txtLogger
        self.totalEpochs = args.trainEpochs
        self.trainingTime = 0
        self.selectedModel = "model" + os.sep + args.model

        self.stepMaxReward = calculateCurricStepMaxReward(ENV_NAMES.ALL_ENVS)

        self.trainingInfoJson = {}
        self.logFilePath = storage.getLogFilePath(["storage", args.model, "status.json"])

        self.curriculaEnvDetails = {}
        self.model = args.model
        self.modelExists = os.path.exists(self.logFilePath)
        self.allEnvsSimultaneous = args.allSimultaneous
        self.latestReward = 0
        self.startEpoch = 1

        if self.modelExists:
            self.loadTrainingInfo()
        else:
            self.trainingInfoJson = RollingHorizon.initTrainingInfo(self.cmdLineString, self.logFilePath, self.seed,
                                                                    self.stepMaxReward, None, self.args)

        self.envNames = self.updateEnvNames(self.envDifficulty)
        self.trainEachCurriculum(self.startEpoch, self.totalEpochs, self.iterationsDone, self.envNames)

    def trainEachCurriculum(self, startEpoch: int, totalEpochs: int, iterationsDone: int, envNames: list):
        assert self.allEnvsSimultaneous
        for epoch in range(startEpoch, totalEpochs):
            iterationsDone = train.startTraining(iterationsDone + self.ITERATIONS_PER_EVALUATE, iterationsDone,
                                                 self.selectedModel, envNames, self.args, self.txtLogger)
            if epoch == 0:
                self.ITERATIONS_PER_EVALUATE = iterationsDone
            reward = evaluate.evaluateAgent(self.selectedModel, self.envDifficulty, self.args, self.txtLogger)

            self.envDifficulty = RollingHorizon.calculateEnvDifficulty(reward, self.stepMaxReward)
            if self.allEnvsSimultaneous:
                envNames = self.updateEnvNames(self.envDifficulty)
            else:
                # [a b c d]. Perform poor -> Go down in difficulty ; perform well -> go up in difficulty; maybe 2/2 split
                pass
            self.updateTrainingInfo(self.trainingInfoJson, epoch, envNames, reward, self.envDifficulty, iterationsDone)
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
        txtLogger.info(
            f"Current rewards after {currentEpoch}: {reward}")
        txtLogger.info(f"\nEPOCH: {epoch} SUCCESS (total: {totalEpochs})\n ")

    @staticmethod
    def updateEnvNames(difficulty) -> list:
        envNames = []
        for j in range(len(ENV_NAMES.ALL_ENVS)):
            index = j
            envNames.append(getEnvFromDifficulty(index, difficulty))
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
        self.envNames = self.trainingInfoJson[curriculaEnvDetailsKey]
        self.startEpoch = self.trainingInfoJson[epochsDone]
        if len(self.trainingInfoJson[snapshotScoreKey]) > 0:
            self.latestReward = self.trainingInfoJson[snapshotScoreKey][-1]
        self.txtLogger.info(f'Reloading state\nContinuing from epoch {self.startEpoch}')
