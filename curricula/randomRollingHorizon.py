import os
import numpy as np
from numpy import ndarray

import utils
from curricula import train, evaluate, RollingHorizon
from utils import getModelWithCandidatePrefix, ENV_NAMES, getEnvFromDifficulty
from utils.curriculumHelper import *


class RandomRollingHorizon(RollingHorizon):
    """
    This class represents a "biased" random rolling horizon. I.e not fully random (like the class
    FullRandomHorizon.
    The difference is that the best curriculum is the base for the next iteration for x% of the curricula in the list
    This aims to achieve that given the last best curriculum, based on this we want to continue most of our training.
    """

    def __init__(self, txtLogger, startTime, cmdLineString: str, args, fullRandom):
        super().__init__(txtLogger, startTime, cmdLineString, args)
        self.fullRandom = fullRandom
        if not self.fullRandom:
            self.curricConsecutivelyChosen = 0
            self.currentBestCurriculum = None
            self.lastChosenCurriculum = None
            # TODO 1/3 probably needs to be updated upon reload
        # self.txtLogger.info(f"curricula list start {self.curricula}")

    def getCurrentBestCurriculum(self):
        currentBestCurriculumIdx = np.argmax(self.currentRewardsDict)
        currentBestCurriculum = self.curricula[currentBestCurriculumIdx]
        if not self.fullRandom:  # TODO 1/3 this is probably very unclean
            self.currentBestCurriculum = currentBestCurriculum
        return currentBestCurriculum

    def getCurrentBestModel(self):
        currentBestCurriculumIdx = np.argmax(self.currentRewardsDict)
        currentBestModel = utils.getModelWithCurricSuffix(self.selectedModel, currentBestCurriculumIdx)
        return currentBestModel

    def updateSpecificInfo(self, epoch) -> None:
        # TODO 1/3 this should be renamed (this was intended for trainingInfoJson updates)
        if self.fullRandom:
            self.curricula = self.randomlyInitializeCurricula(self.numCurric, self.stepsPerCurric, self.envDifficulty,
                                                              self.paraEnvs, self.seed + epoch)
        else:
            self.curricula = self.updateCurriculaAfterHorizon(self.lastChosenCurriculum, self.numCurric,
                                                              self.envDifficulty)
            self.curricConsecutivelyChosen = \
                self.calculateConsecutivelyChosen(self.curricConsecutivelyChosen, self.currentBestCurriculum,
                                                  self.lastChosenCurriculum)

    def executeOneEpoch(self, epoch: int) -> None:
        currentRewards = {"curric_" + str(i): [] for i in range(len(self.curricula))}
        snapshotRewards = {"curric_" + str(i): [] for i in range(len(self.curricula))}
        for i in range(len(self.curricula)):
            reward = self.trainACurriculum(i, self.iterationsDone, -1, self.curricula)
            # TODO tet this so that the changes did not break everything
            currentRewards["curric_" + str(i)] = np.sum(reward)
            snapshotRewards["curric_" + str(i)] = reward[0]
            # self.txtLogger.info(f"\tepoch {epoch }: RRH Curriculum {i} done")

        self.currentRewardsDict = currentRewards
        self.currentSnapshotRewards = snapshotRewards
        self.curriculaEnvDetails = self.curricula
        # self.txtLogger.info(f"\n\tcurrentRewards for : {self.currentRewardsDict}")
        # self.txtLogger.info(f"\tsnapshot Rewards for : {self.currentSnapshotRewards}")
        # self.txtLogger.info(f"\tcurrentEnvDetails for : {self.curriculaEnvDetails}\n\n")

    def calculateConsecutivelyChosen(self, consecutiveCount, currentBestCurriculum, lastChosenCurriculum) -> int:
        self.lastChosenCurriculum = currentBestCurriculum
        if consecutiveCount + 1 < len(self.curricula[0]) and \
                (currentBestCurriculum == lastChosenCurriculum or lastChosenCurriculum is None):
            return consecutiveCount + 1
        return 0

    def updateCurriculaAfterHorizon(self, bestCurriculum: list, numberOfCurricula: int, envDifficulty: int) -> list:
        """
        Updates the List of curricula by using the last N-1 Envs, and randomly selecting a last new one
        :param envDifficulty:
        :param numberOfCurricula:
        :param bestCurriculum: full env list of the curriculum that performed best during last epoch
                (i.e. needs to be cut by 1 element!)
        """
        curricula = []
        # TODO 1/3 maybe only do this for a percentage of curricula, and randomly set the others OR instead of using [1:], use [1:__]
        # TODO 1/3 test this
        for i in range(numberOfCurricula):
            curricula.append(bestCurriculum[1:])
            envId = np.random.randint(0, len(envList))
            curricula[i].append(getEnvFromDifficulty(envId, envList, envDifficulty))
        assert len(curricula) == numberOfCurricula
        return curricula

    def getCurriculumName(self, i, genNr):
        assert genNr == -1, "this parameter shouldnt matter for RRH"
        return utils.getModelWithCurricSuffix(self.selectedModel, i)
