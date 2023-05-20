import os
import random
from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray

import utils
from curricula import train, evaluate
from utils import ENV_NAMES, getEnvFromDifficulty
from utils import getModelWithCandidatePrefix
from utils.curriculumHelper import *


class RollingHorizon(ABC):
    def __init__(self, txtLogger, startTime, cmdLineString: str, args):
        random.seed(args.seed)
        self.args = args
        self.numCurric = args.numCurric
        self.stepsPerCurric = args.stepsPerCurric
        self.cmdLineString = cmdLineString
        self.lastEpochStartTime = startTime
        self.envDifficulty = 0
        self.exactIterationsSet = False
        self.seed = args.seed
        self.paraEnvs = args.paraEnv
        # TODO does RHEA even need a curric list becasue it gets generated always anyway
        self.curricula = self.randomlyInitializeCurricula(args.numCurric, args.stepsPerCurric, self.envDifficulty,
                                                          self.paraEnvs, self.seed)

        self.ITERATIONS_PER_ENV = args.iterPerEnv
        self.iterationsDone = 0
        self.txtLogger = txtLogger
        self.selectedModel = utils.getEpochModelName(args.model, 0)
        self.totalEpochs = args.trainEpochs
        self.trainingTime = 0

        self.stepMaxReward = calculateCurricStepMaxReward(len(ENV_NAMES.ALL_ENVS))
        self.curricMaxReward = calculateCurricMaxReward(self.stepsPerCurric, self.stepMaxReward, args.gamma)

        self.trainingInfoJson = {}
        self.logFilePath = os.getcwd() + "\\storage\\" + args.model + "\\status.json"  # TODO maybe outsource
        self.gamma = args.gamma  # TODO is gamma used properly? Do RH -> Get Max thingy, and update difficulty based on the RH reward or snapshot reward?
        self.currentRewardsDict = {}
        self.currentSnapshotRewards = {}
        self.curriculaEnvDetails = {}
        self.model = args.model
        self.modelExists = os.path.exists(self.logFilePath)
        self.txtLogger.info(f"curricula list start {self.curricula}")

    def saveFirstStepOfModel(self, exactIterationsPerEnv: int, nameOfCurriculumI: str):
        if not self.exactIterationsSet:  # TODO refactor this to common method
            self.exactIterationsSet = True
            self.ITERATIONS_PER_ENV = exactIterationsPerEnv - 1
        utils.copyAgent(src=nameOfCurriculumI, dest=utils.getModelWithCandidatePrefix(
            nameOfCurriculumI), txtLogger=self.txtLogger)  # save TEST_e1_curric0 -> + _CANDIDATE
        self.txtLogger.info(f"ITERATIONS PER ENV = {self.ITERATIONS_PER_ENV}")
        self.trainingInfoJson[iterationsPerEnvKey] = self.ITERATIONS_PER_ENV

    # TODO move curriculumHelper methods here

    def startCurriculumTraining(self):
        """
        Starts the training loop
        """
        startEpoch, rewards = self.initializeTrainingVariables(self.modelExists)
        for epoch in range(startEpoch, self.totalEpochs):
            self.txtLogger.info(
                f"\n--------------------------------------------------------------\n                     START EPOCH {epoch}\n--------------------------------------------------------------\n")
            self.selectedModel = utils.getEpochModelName(self.model, epoch)

            self.executeOneEpoch(epoch)

            self.iterationsDone += self.ITERATIONS_PER_ENV
            nextModel = utils.getEpochModelName(self.model, epoch + 1)
            rewards["epoch" + str(epoch)] = self.currentRewardsDict

            # normalize currentRewards
            currentRewardsList = [self.currentRewardsDict[key] / self.curricMaxReward for key in
                                  self.currentRewardsDict]
            bestCurriculumScore: float = np.max(currentRewardsList)
            currentSnapshotScore: float = np.max(list(self.currentSnapshotRewards.values()))

            currentBestModel = self.getCurrentBestModel()
            currentBestCurriculum = self.getCurrentBestCurriculum()
            utils.copyAgent(src=getModelWithCandidatePrefix(currentBestModel), dest=nextModel, txtLogger=self.txtLogger)

            self.envDifficulty = self.calculateEnvDifficulty(currentSnapshotScore, self.stepMaxReward)
            self.updateTrainingInfo(self.trainingInfoJson, epoch, currentBestCurriculum, rewards, bestCurriculumScore,
                                    currentSnapshotScore, self.iterationsDone, self.envDifficulty,
                                    self.lastEpochStartTime, self.curricula, self.curriculaEnvDetails, self.logFilePath)
            # TODO the self.curricula call will fail for RRH becuase its been updated arleady
            self.updateSpecificInfo(epoch)
            self.logInfoAfterEpoch(epoch, currentBestCurriculum, bestCurriculumScore, currentSnapshotScore,
                                   self.trainingInfoJson, self.txtLogger, self.stepMaxReward, self.totalEpochs)

            self.resetEpochVariables()

        printFinalLogs(self.trainingInfoJson, self.txtLogger)

    def trainEachCurriculum(self, i: int, iterationsDone: int, genNr: int, curricula) -> ndarray:
        """
        Simulates a horizon and returns the rewards obtained after evaluating the state at the end of the horizon
        """
        # TODO can probably remove genNr from methodparam
        reward = np.zeros(len(curricula[i])) # todo [i] vs not
        # Save epoch_X -> epoch_X_curricI_genJ
        nameOfCurriculumI = self.getCurriculumName(i, genNr)
        utils.copyAgent(src=self.selectedModel, dest=nameOfCurriculumI, txtLogger=self.txtLogger)
        initialIterationsDone = iterationsDone
        for j in range(len(curricula[i])):
            iterationsDone = train.startTraining(iterationsDone + self.ITERATIONS_PER_ENV, iterationsDone,
                                                 nameOfCurriculumI, curricula[i][j], self.args, self.txtLogger)
            #reward[j] = ((self.gamma ** j) * evaluate.evaluateAgent(nameOfCurriculumI, self.envDifficulty, self.args,
             #                                                       self.txtLogger))
            self.txtLogger.info(f"\tIterations Done {iterationsDone}")
            if j == 0:
                self.saveFirstStepOfModel(iterationsDone - initialIterationsDone, nameOfCurriculumI)  # TODO testfor ep0
            self.txtLogger.info(f"\tTrained iteration j={j} of curriculum {nameOfCurriculumI}")
            self.txtLogger.info(f"\tReward for curriculum {nameOfCurriculumI} = {reward} (1 entry = 1 env)\n\n")
            self.txtLogger.info("-------------------------------")
        return reward

    @abstractmethod
    def executeOneEpoch(self, epoch: int) -> None:
        pass # TODO is epoch used ?

    @abstractmethod
    def updateSpecificInfo(self, epoch) -> None:
        pass

    @abstractmethod
    def getCurrentBestModel(self):
        """
        Gets the name of the currently best performing curriculum after the horizons were rolled out
        :return:
        """
        pass

    @abstractmethod
    def getCurrentBestCurriculum(self):
        """
        Gets the env list of the currently best performing curriculum after the horizons were rolled out
        :return:
        """
        pass

    def resetEpochVariables(self) -> None:
        self.currentRewardsDict = {}
        self.currentSnapshotRewards = {}
        self.curriculaEnvDetails = {}
        self.lastEpochStartTime = datetime.now()

    def initializeTrainingVariables(self, modelExists) -> tuple:
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
            """
            if isinstance(self, RandomRollingHorizon):
                if not self.fullRandom:
                    self.curricConsecutivelyChosen = self.trainingInfoJson[consecutivelyChosen]
                    self.lastChosenCurriculum = self.trainingInfoJson[""] # TODO
            """
            self.lastEpochStartTime = self.trainingInfoJson["startTime"]  # TODO use right keys

            # TODO test rewardsDictb ecasue RRH has "rewards" with gets reutrned

            # delete existing folders, that were created ---> maybe just last one because others should be finished ...
            # TODO maybe do the deletion automatically, but it doesnt matter
            """
            for k in range(self.numCurric):
                path = utils.getModelWithCurricSuffix(self.model, startEpoch, k)
                if utils.deleteModelIfExists(path):
                    print("deleted", k)
                    snapshotPath = utils.getModelWithCandidatePrefix(path)
                    utils.deleteModelIfExists(snapshotPath)
                    delete Gen thingy if need be
                    # TODO test if delete _gen folders; OR probably get prefix -> look for them in list, delete all of these folders that contain it
                else:
                    self.txtLogger.info(f"Nothing to delete {k}")
                    break
            """
            assert len(self.curricula) == self.trainingInfoJson["curriculaEnvDetails"]["epoch0"]  # TODO ?
            self.txtLogger.info(f"Continung training from epoch {startEpoch}... [total epochs: {self.totalEpochs}]")
        else:
            self.txtLogger.info("Creating model. . .")
            train.startTraining(0, 0, self.selectedModel, [getEnvFromDifficulty(0, self.envDifficulty)], self.args,
                                self.txtLogger)
            self.trainingInfoJson = self.initTrainingInfo(self.cmdLineString, self.logFilePath, self.seed, self.args)
            startEpoch = 1
            utils.copyAgent(src=self.selectedModel, dest=utils.getEpochModelName(self.model, startEpoch),
                            txtLogger=self.txtLogger)  # copy epoch0 -> epoch1
            self.txtLogger.info(f"\nThe training will go on for {self.totalEpochs} epochs\n")
            rewardsDict = {}

            self.curricula = self.randomlyInitializeCurricula(self.numCurric, self.stepsPerCurric, self.envDifficulty,
                                                              self.paraEnvs, self.seed)
            train.startTraining(0, 0, self.selectedModel, [getEnvFromDifficulty(0, self.envDifficulty)], self.args,
                                self.txtLogger)

        return startEpoch, rewardsDict

    def evaluateCurriculumResults(self, evaluationDictionary):
        # evaluationDictionary["actualPerformance"][0] ---> zeigt den avg reward des models zu jedem übernommenen Snapshot
        # evaluationDictionary["actualPerformance"][1] ---> zeigt die zuletzt benutzte Umgebung zu dem Zeitpunkt an
        #
        tmp = []
        i = 0
        for reward, env in tmp:
            i += 1

        # Dann wollen wir sehen, wie das curriculum zu dem jeweiligen zeitpunkt ausgesehen hat.
        # # Aber warum? Und wie will man das nach 20+ durchläufen plotten

    def initTrainingInfo(self, cmdLineString, logFilePath, seed, args) -> dict:
        """
        Initializes the trainingInfo dictionary
        :return:
        """
        trainingInfoJson = {selectedEnvs: [],
                            bestCurriculas: [],
                            curriculaEnvDetailsKey: {},
                            rewardsKey: {},
                            actualPerformance: {},
                            epochsDone: 1,
                            epochTrainingTime: [],
                            snapshotScoreKey: [],
                            sumTrainingTime: 0,
                            cmdLineStringKey: cmdLineString,
                            difficultyKey: [0],
                            seedKey: seed,
                            iterationsPerEnvKey: args.iterPerEnv,
                            consecutivelyChosen: 0,
                            fullArgs: args,
                            additionalNotes: "",
                            numFrames: 0}
        saveTrainingInfoToFile(logFilePath, trainingInfoJson)
        return trainingInfoJson

    def logInfoAfterEpoch(self, epoch, currentBestCurriculum, bestReward, snapshotReward, trainingInfoJson, txtLogger,
                          maxReward, totalEpochs):
        """
        Logs relevant training info after a training epoch is done and the trainingInfo was updated
        :param snapshotReward:
        :param maxReward:
        :param txtLogger:
        :param trainingInfoJson:
        :param totalEpochs:
        :param epoch:
        :param currentBestCurriculum: the id of the current best curriculum
        :param bestReward:
        :return:
        """
        selectedEnv = trainingInfoJson[selectedEnvs][-1]

        txtLogger.info(
            f"Best results in epoch {epoch} came from curriculum {currentBestCurriculum}")
        txtLogger.info(
            f"CurriculaEnvDetails {curriculaEnvDetailsKey}; selectedEnv: {selectedEnv}")
        txtLogger.info(f"Reward of best curriculum: {bestReward}. \
            Snapshot Reward {snapshotReward}. That is {bestReward / maxReward} of maxReward")

        txtLogger.info(f"\nEPOCH: {epoch} SUCCESS (total: {totalEpochs})\n ")

    def calculateEnvDifficulty(self, currentReward, maxReward) -> int:
        # TODO EXPERIMENT: that is why i probably should have saved the snapshot reward
        if currentReward < maxReward * .25:
            return 0
        elif currentReward < maxReward * .75:
            return 1
        return 2

    def randomlyInitializeCurricula(self, numberOfCurricula: int, stepsPerCurric: int, envDifficulty: int, paraEnv: int,
                                    seed: int) -> list:
        """
        Initializes list of curricula randomly. Allows duplicates, but they are extremely unlikely.
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

    def updateTrainingInfo(self, trainingInfoJson, epoch: int, bestCurriculum: list, fullRewardsDict,
                           currentScore: float, snapshotScore: float, framesDone, envDifficulty: int,
                           lastEpochStartTime, curricula, curriculaEnvDetails, logFilePath) -> None:
        """
        Updates the training info dictionary
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
        :param currentScore: the current best score
        """
        trainingInfoJson[epochsDone] = epoch + 1
        trainingInfoJson[numFrames] = framesDone
        trainingInfoJson[snapshotScoreKey].append(snapshotScore)

        trainingInfoJson[selectedEnvs].append(bestCurriculum[0])
        trainingInfoJson[bestCurriculas].append(bestCurriculum)
        trainingInfoJson[rewardsKey] = fullRewardsDict
        trainingInfoJson[actualPerformance]["epoch_" + str(epoch)] = \
            {"curricScore": currentScore, "snapshotScore": snapshotScore, "curriculum": bestCurriculum}
        trainingInfoJson[curriculaEnvDetailsKey]["epoch_" + str(epoch)] = curriculaEnvDetails
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
