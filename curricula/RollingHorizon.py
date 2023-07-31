import argparse
import os
import random
import shutil
from abc import ABC, abstractmethod

from numpy import ndarray

import utils
from curricula import train, evaluate, RollingHorizonEvolutionaryAlgorithm
from utils import ENV_NAMES, getEnvFromDifficulty, storage
from utils import getModelWithCandidatePrefix
from utils.curriculumHelper import *


class RollingHorizon(ABC):
    def __init__(self, txtLogger, startTime, cmdLineString: str, args: argparse.Namespace):
        random.seed(args.seed)
        self.args = args
        self.numCurric = args.numCurric
        self.stepsPerCurric = args.stepsPerCurric
        self.cmdLineString = cmdLineString
        self.lastEpochStartTime = startTime
        self.envDifficulty: float = 1.0  # 1 meaning 100% maxsteps allowed
        self.exactIterationsSet = False
        self.seed = args.seed
        self.paraEnvs = args.paraEnv
        self.difficultyStepsize = args.difficultyStepsize
        self.curricula = self.randomlyInitializeCurricula(args.numCurric, args.stepsPerCurric, self.envDifficulty,
                                                          self.paraEnvs, self.seed)

        self.ITERATIONS_PER_ENV = args.iterPerEnv
        self.iterationsDone = 0
        self.txtLogger = txtLogger
        self.model = args.model + "_s" + str(self.seed)
        self.selectedModel = utils.getEpochModelName(self.model, 0)
        self.totalEpochs = 5000000 // self.ITERATIONS_PER_ENV + 1  # TODO remove args.trainEpochs
        # TODO calculate totalEpochs based on another args parameter (4kk vs 5kk etc)
        # TODO remove --model param and let it be created automatically (or optional for debug)
        self.trainingTime = 0

        self.stepMaxReward = calculateCurricStepMaxReward(ENV_NAMES.ALL_ENVS, args.noRewardShaping)
        self.curricMaxReward = calculateCurricMaxReward(self.stepsPerCurric, self.stepMaxReward, args.gamma)
        self.trainingInfoJson = {}
        self.logFilePath = storage.getLogFilePath(["storage", self.model, "status.json"])
        self.gamma = args.gamma
        self.currentRewardsDict = {}
        self.currentSnapshotRewards = {}
        self.curriculaEnvDetails = {}
        self.modelExists = os.path.exists(self.logFilePath)

    def saveFirstStepOfModel(self, exactIterationsPerEnv: int, nameOfCurriculumI: str):
        if not self.exactIterationsSet:
            self.exactIterationsSet = True
            self.ITERATIONS_PER_ENV = exactIterationsPerEnv - 1
        utils.copyAgent(src=nameOfCurriculumI, dest=utils.getModelWithCandidatePrefix(
            nameOfCurriculumI), txtLogger=self.txtLogger)  # save TEST_e1_curric0 -> + _CANDIDATE
        self.trainingInfoJson[iterationsPerEnvKey] = self.ITERATIONS_PER_ENV

    def startCurriculumTraining(self):
        """
        Starts the training loop
        """
        startEpoch, rewards = self.initializeTrainingVariables(self.modelExists)
        for epoch in range(startEpoch, self.totalEpochs):
            self.updateModelName(epoch)

            self.executeOneEpoch(epoch)

            self.iterationsDone += self.ITERATIONS_PER_ENV
            nextModel = utils.getEpochModelName(self.model, epoch + 1)
            rewards["epoch" + str(epoch)] = self.currentRewardsDict

            # normalize currentRewards
            print("currentRewards", self.currentRewardsDict)
            currentRewardsList = [self.currentRewardsDict[key] for key in self.currentRewardsDict]
            bestCurricScoreRaw: float = np.max(currentRewardsList)
            currentSnapshotScore: float = np.max(list(self.currentSnapshotRewards.values()))
            currentBestModel = self.getCurrentBestModel()
            currentBestCurriculum = self.getCurrentBestCurriculum()
            utils.copyAgent(src=getModelWithCandidatePrefix(currentBestModel), dest=nextModel, txtLogger=self.txtLogger)

            self.envDifficulty = calculateEnvDifficulty(self.iterationsDone, self.difficultyStepsize)
            self.updateTrainingInfo(self.trainingInfoJson, epoch, currentBestCurriculum, rewards, bestCurricScoreRaw,
                                    currentSnapshotScore,
                                    self.iterationsDone, self.envDifficulty, self.lastEpochStartTime, self.curricula,
                                    self.curriculaEnvDetails,
                                    self.logFilePath, self.curricMaxReward)
            self.updateSpecificInfo(
                epoch)  # TODO this should probably be already done in the executeOneEpoch method, name is confusing
            self.logInfoAfterEpoch(epoch, currentBestCurriculum, bestCurricScoreRaw, currentSnapshotScore,
                                   self.trainingInfoJson, self.txtLogger,
                                   self.stepMaxReward, self.totalEpochs)

            self.resetEpochVariables()
        self.trainingInfoJson["done"] = True
        saveTrainingInfoToFile(self.logFilePath, self.trainingInfoJson)
        printFinalLogs(self.trainingInfoJson, self.txtLogger)

    def trainACurriculum(self, i: int, iterationsDone: int, genNr: int, curricula: list) -> ndarray:
        """
        Simulates a horizon and returns the rewards obtained after evaluating the state at the end of the horizon
        """
        # TODO can probably remove genNr from method param
        isMultiObj = False
        reward = np.zeros(len(curricula[i]))
        # if isinstance(self, type(RollingHorizonEvolutionaryAlgorithm)):
        if hasattr(self, "multiObj") and self.multiObj: # TODO this is very ugly, but isinstance of did not work
            isMultiObj = True
            if isMultiObj:
                reward = np.zeros((len(curricula[i]), self.objectives))
        nameOfCurriculumI = self.getCurriculumName(i, genNr)
        utils.copyAgent(src=self.selectedModel, dest=nameOfCurriculumI, txtLogger=self.txtLogger)
        initialIterationsDone = iterationsDone
        for j in range(len(curricula[i])):
            iterationsDone = train.startTraining(iterationsDone + self.ITERATIONS_PER_ENV, iterationsDone,
                                                 nameOfCurriculumI, curricula[i][j],
                                                 self.args, self.txtLogger)
            tmp = evaluate.evaluateAgent(nameOfCurriculumI, self.envDifficulty, self.args, self.txtLogger)
            if isMultiObj:
                for k in range(len(tmp)):
                    reward[j][k] = (self.gamma ** j) * tmp[k]
            else:
                reward[j] = (self.gamma ** j) * np.sum(tmp)
            if j == 0:
                self.saveFirstStepOfModel(iterationsDone - initialIterationsDone, nameOfCurriculumI)
            self.logInfoAfterCurriculum(nameOfCurriculumI, iterationsDone, reward, j)
        return reward

    def logInfoAfterCurriculum(self, nameOfCurriculumI, iterationsDone, rewardList, j):
        # self.txtLogger.info(
        #   f"\tTrained iteration j={j} of curriculum {nameOfCurriculumI}. Iterations done {iterationsDone}")
        # self.txtLogger.info(f"\tReward for curriculum {nameOfCurriculumI} = {rewardList} (1 entry = 1 curric step)")
        currentMax = (self.gamma ** j) * self.stepMaxReward  # TODO fix
        # self.txtLogger.info(f"\tReward-%-Performance {rewardList / currentMax}\n\n")
        # self.txtLogger.info("-------------------------------")

    def resetEpochVariables(self) -> None:
        self.currentRewardsDict = {}
        self.currentSnapshotRewards = {}
        self.curriculaEnvDetails = {}
        self.lastEpochStartTime = datetime.now()

    def initializeTrainingVariables(self, modelExists: bool) -> tuple:
        """
        Initializes and returns all the necessary training variables
        :param modelExists: whether the path to the model already exists or not
        """
        if modelExists:
            with open(self.logFilePath, 'r') as f:
                self.trainingInfoJson = json.loads(f.read())

            self.iterationsDone = self.trainingInfoJson[numFrames]
            startEpoch = self.trainingInfoJson[epochsDone]
            if iterationsPerEnvKey in self.trainingInfoJson:
                self.ITERATIONS_PER_ENV = self.trainingInfoJson[iterationsPerEnvKey]
            rewardsDict = self.trainingInfoJson[rewardsKey]
            self.seed = self.trainingInfoJson[seedKey]
            self.envDifficulty = self.trainingInfoJson[difficultyKey][-1]
            self.ITERATIONS_PER_ENV = self.trainingInfoJson[iterationsPerEnvKey]
            self.exactIterationsSet = True

            register(
                id=ENV_NAMES.DOORKEY_12x12 + ENV_NAMES.CUSTOM_POSTFIX + str(self.envDifficulty),
                entry_point="minigrid.envs:DoorKeyEnv",
                kwargs={"size": 12, "max_steps": int(maxStepsEnv4 * self.envDifficulty)},
            )
            register(
                id=ENV_NAMES.DOORKEY_10x10 + ENV_NAMES.CUSTOM_POSTFIX + str(self.envDifficulty),
                entry_point="minigrid.envs:DoorKeyEnv",
                kwargs={"size": 10, "max_steps": int(maxStepsEnv4 * self.envDifficulty)},
            )

            register(
                id=ENV_NAMES.DOORKEY_8x8 + ENV_NAMES.CUSTOM_POSTFIX + str(self.envDifficulty),
                entry_point="minigrid.envs:DoorKeyEnv",
                kwargs={"size": 8, "max_steps": int(maxStepsEnv4 * self.envDifficulty)},
            )

            register(
                id=ENV_NAMES.DOORKEY_6x6 + ENV_NAMES.CUSTOM_POSTFIX + str(self.envDifficulty),
                entry_point="minigrid.envs:DoorKeyEnv",
                kwargs={"size": 6, "max_steps": int(maxStepsEnv4 * self.envDifficulty)},
            )
            # TODO move registration stuff

            # TODO maybe move this to storage
            prefix = f"epoch_{startEpoch}_"
            path = os.getcwd() + os.sep + "storage" + os.sep + self.model + os.sep
            dirs = (os.listdir(path))
            for directory in dirs:
                if prefix in directory:
                    try:
                        shutil.rmtree(path + os.sep + directory)
                        print("deleted: ./", directory)
                    except Exception as e:
                        print("Exception while deleting:", e)

            self.txtLogger.info(
                f"Continung training from epoch {startEpoch}... [total epochs: {self.totalEpochs}], seed: {self.seed}")
        else:
            self.txtLogger.info("Creating model. . .")
            train.startTraining(0, 0, self.selectedModel, [getEnvFromDifficulty(0, self.envDifficulty)], self.args,
                                self.txtLogger)
            self.trainingInfoJson = self.initTrainingInfo(self.cmdLineString, self.logFilePath, self.seed,
                                                          self.stepMaxReward, self.curricMaxReward, self.args)
            startEpoch = 1
            utils.copyAgent(src=self.selectedModel, dest=utils.getEpochModelName(self.model, startEpoch),
                            txtLogger=self.txtLogger)  # copy epoch0 -> epoch1
            self.txtLogger.info(f"\nThe training will go on for {self.totalEpochs} epochs, seed: {self.seed}\n")
            rewardsDict = {}

            self.curricula = self.randomlyInitializeCurricula(self.numCurric, self.stepsPerCurric, self.envDifficulty,
                                                              self.paraEnvs, self.seed)
        return startEpoch, rewardsDict

    @staticmethod
    def initTrainingInfo(cmdLineString: str, logFilePath: str, seed: int, stepMaxReward: float, curricMaxReward: float,
                         args: argparse.Namespace) -> dict:
        """
        Initializes the trainingInfo dictionary
        :return:
        """
        trainingInfoJson = {selectedEnvs: {},
                            bestCurriculas: {},
                            curriculaEnvDetailsKey: {},
                            rewardsKey: {},
                            actualPerformance: {},
                            maxStepRewardKey: stepMaxReward,
                            maxCurricRewardKey: curricMaxReward,
                            epochsDone: 1,
                            "done": False,
                            epochTrainingTime: [],
                            snapshotScoreKey: [],
                            sumTrainingTime: 0,
                            cmdLineStringKey: cmdLineString,
                            difficultyKey: [1.0],
                            seedKey: seed,
                            iterationsPerEnvKey: args.iterPerEnv,
                            consecutivelyChosen: 0,
                            fullArgs: args,
                            usedEnvEnumerationKey: ENV_NAMES.ALL_ENVS,
                            additionalNotes: "",
                            numFrames: 0}
        saveTrainingInfoToFile(logFilePath, trainingInfoJson)
        return trainingInfoJson

    @staticmethod
    def logInfoAfterEpoch(epoch, currentBestCurriculum, bestReward, snapshotReward, trainingInfoJson, txtLogger,
                          stepMaxReward, totalEpochs):
        """
        Logs relevant training info after a training epoch is done and the trainingInfo was updated
        :param stepMaxReward: the highest reward possible for the first step of a curriculum
        :param snapshotReward: the reward of the first step of the best performing curriulum from the last epoch
        :param txtLogger: the reference to the logger which logs all the info happening during training
        :param trainingInfoJson: the dictionary which stores all the relevant info of the performance of the model
        :param totalEpochs: the amount of epochs the training will go on for
        :param epoch: the current epoch nr
        :param currentBestCurriculum: the id of the current best curriculum
        :param bestReward: the reward of the best performing curriculum from the last epoch
        :return:
        """
        currentEpoch = "epoch_" + str(epoch)
        selectedEnv = trainingInfoJson[selectedEnvs][currentEpoch]

        txtLogger.info(
            f"Best results in epoch {epoch} came from curriculum {currentBestCurriculum}")
        envDetails = trainingInfoJson[curriculaEnvDetailsKey][currentEpoch]
        # txtLogger.info(
        #   f"CurriculaEnvDetails {envDetails}; selectedEnv: {selectedEnv}")
        txtLogger.info(f"Raw Reward of best curriculum: {bestReward}. \
            Snapshot Reward {snapshotReward}. That is {round(snapshotReward / stepMaxReward, 3)} of maxReward")

        txtLogger.info(f"\nEPOCH: {epoch} SUCCESS (total: {totalEpochs})\n ")

    @staticmethod
    def randomlyInitializeCurricula(numberOfCurricula: int, stepsPerCurric: int, envDifficulty: int, paraEnv: int,
                                    seed: int) -> list:
        """
        Initializes list of curricula randomly. Method allows duplicates, but they are extremely unlikely.
        :param paraEnv: the amount of envs that will be trained in parallel per step of a curriculum
        :param seed: the random seed
        :param envDifficulty:
        :param numberOfCurricula: how many curricula will be generated
        :param stepsPerCurric: how many steps a curriculum contains
        """
        random.seed(seed)
        curricula = []
        for i in range(numberOfCurricula):
            current = []
            for j in range(stepsPerCurric):
                indices = random.choices(range(len(ENV_NAMES.ALL_ENVS)), k=paraEnv)
                newCurriculum = [getEnvFromDifficulty(idx, envDifficulty) for idx in indices]
                current.append(newCurriculum)
            curricula.append(current)
        assert len(curricula) == numberOfCurricula
        assert len(curricula[0]) == stepsPerCurric
        return curricula

    @staticmethod
    def updateTrainingInfo(trainingInfoJson, epoch: int, bestCurriculum: list, fullRewardsDict,
                           currentScoreRaw: float, snapshotScore: float, framesDone, envDifficulty: int,
                           lastEpochStartTime, curricula, curriculaEnvDetails, logFilePath, curricMaxReward) -> None:
        """
        Updates the training info dictionary
        :param curricMaxReward:
        :param snapshotScore:
        :param curriculaEnvDetails:
        :param logFilePath:
        :param curricula:
        :param lastEpochStartTime:
        :param envDifficulty:
        :param framesDone:
        :param trainingInfoJson:
        :param epoch: current epoch
        :param bestCurriculum: the curriculum that had the highest reward in the latest epoch
        :param fullRewardsDict: the dict of rewards for each generation and each curriculum
        :param currentScoreRaw: the current best score
        """
        currentEpoch = "epoch_" + str(epoch)

        trainingInfoJson[epochsDone] = epoch + 1
        trainingInfoJson[numFrames] = framesDone
        trainingInfoJson[snapshotScoreKey].append(snapshotScore)

        trainingInfoJson[selectedEnvs][currentEpoch] = bestCurriculum[0]
        trainingInfoJson[bestCurriculas][currentEpoch] = bestCurriculum
        trainingInfoJson[rewardsKey] = fullRewardsDict
        trainingInfoJson[actualPerformance][currentEpoch] = \
            {"curricScoreRaw": currentScoreRaw,
             "curricScoreNormalized": currentScoreRaw / curricMaxReward,
             "snapshotScoreRaw": snapshotScore, "curriculum": bestCurriculum}
        trainingInfoJson[curriculaEnvDetailsKey][currentEpoch] = curriculaEnvDetails
        trainingInfoJson[difficultyKey].append(envDifficulty)

        now = datetime.now()
        timeSinceLastEpoch = (now - lastEpochStartTime).total_seconds()
        trainingInfoJson[epochTrainingTime].append(timeSinceLastEpoch)
        trainingInfoJson[sumTrainingTime] += timeSinceLastEpoch

        # Debug Logs
        trainingInfoJson["currentListOfCurricula"] = curricula

        saveTrainingInfoToFile(logFilePath, trainingInfoJson)
        # TODO how expensive is it to always overwrite everything?

    @abstractmethod
    def getCurriculumName(self, i, genNr):
        pass

    @abstractmethod
    def executeOneEpoch(self, epoch: int) -> None:
        pass

    @abstractmethod
    def updateSpecificInfo(self, epoch) -> None:
        pass

    @abstractmethod
    def getCurrentBestModel(self) -> str:
        """
        Gets the name of the currently best performing curriculum after the horizons were rolled out
        :return:
        """
        pass

    @abstractmethod
    def getCurrentBestCurriculum(self) -> list:
        """
        Gets the env list of the currently best performing curriculum after the horizons were rolled out
        :return:
        """
        pass

    def updateModelName(self, epoch: int) -> None:
        # self.txtLogger.info(
        #   f"\n--------------------------------------------------------------\n                     START EPOCH {epoch}\n--------------------------------------------------------------\n")
        self.selectedModel = utils.getEpochModelName(self.model, epoch)
