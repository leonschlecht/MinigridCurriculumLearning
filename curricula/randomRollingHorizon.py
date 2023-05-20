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
            # TODO probably needs to be updated upon reload

    def getCurrentBestCurriculum(self):
        currentBestCurriculumIdx = np.argmax(self.currentRewardsDict)
        currentBestCurriculum = self.curricula[currentBestCurriculumIdx]
        if not self.fullRandom:  # TODO this is probably very unclean
            self.currentBestCurriculum = currentBestCurriculum
        return currentBestCurriculum

    def getCurrentBestModel(self):
        currentBestCurriculumIdx = np.argmax(self.currentRewardsDict)
        currentBestModel = utils.getModelWithCurricSuffix(self.selectedModel, currentBestCurriculumIdx)
        return currentBestModel

    def updateSpecificInfo(self) -> None:
        # TODO this should be renamed (this was intended for trainingInfoJson updates)
        if not self.fullRandom:
            self.curricConsecutivelyChosen = \
                self.calculateConsecutivelyChosen(self.curricConsecutivelyChosen, self.currentBestCurriculum,
                                                  self.lastChosenCurriculum)

    def executeOneEpoch(self, epoch: int) -> None:
        currentRewards = {"curric_" + str(i): [] for i in range(len(self.curricula))}
        snapshotRewards = {"curric_" + str(i): [] for i in range(len(self.curricula))}
        for i in range(len(self.curricula)):
            reward = self.trainEachCurriculum(i, self.iterationsDone, 0, self.curricula)
            currentRewards["curric_" + str(i)] = np.sum(reward)
            snapshotRewards["curric_" + str(i)] = reward[0]

        self.currentRewardsDict = currentRewards
        self.currentSnapshotRewards = snapshotRewards
        self.curriculaEnvDetails = self.curricula
        self.txtLogger.info(f"currentRewards for : {self.currentRewardsDict}")
        self.txtLogger.info(f"snapshot Rewards for : {self.currentSnapshotRewards}")
        self.txtLogger.info(f"currentEnvDetails for : {self.curriculaEnvDetails}")
        if self.fullRandom:
            self.curricula = self.randomlyInitializeCurricula(self.numCurric, self.stepsPerCurric, self.envDifficulty,
                                                              self.paraEnvs, self.seed + epoch)
        else:
            self.curricula = self.updateCurriculaAfterHorizon(self.lastChosenCurriculum, self.numCurric,
                                                              self.envDifficulty)

            # TODO the order of updateCurriculaAfterHorizon and updateConsec is not clear

    def trainEachCurriculum(self, i: int, iterationsDone: int, genNr: int, curricula) -> ndarray:
        """
        Simulates a horizon and returns the rewards obtained after evaluating the state at the end of the horizon
        """
        nameOfCurriculumI = utils.getModelWithCurricSuffix(self.selectedModel, i)  # Save TEST_e1 --> TEST_e1_curric0
        rewards = np.zeros(len(self.curricula))
        utils.copyAgent(src=self.selectedModel, dest=nameOfCurriculumI, txtLogger=self.txtLogger)
        initialIterationsDone = iterationsDone
        print("curricula = ", self.curricula[i])
        for j in range(len(self.curricula[i])):
            iterationsDone = train.startTraining(iterationsDone + self.ITERATIONS_PER_ENV, iterationsDone,
                                                 nameOfCurriculumI, self.curricula[i][j], self.args, self.txtLogger)
            self.txtLogger.info(f"Iterations Done {iterationsDone}")
            if j == 0:
                self.saveFirstStepOfModel(iterationsDone - initialIterationsDone, nameOfCurriculumI)
            self.txtLogger.info(f"Trained iteration j={j} of curriculum {nameOfCurriculumI} ")
            #rewards[j] = ((self.gamma ** j) * evaluate.evaluateAgent(nameOfCurriculumI, self.envDifficulty, self.args,
             #                                                        self.txtLogger))

        self.txtLogger.info(f"Rewards for curriculum {nameOfCurriculumI} = {rewards}")
        return rewards

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
        # TODO remove duplicates
        # TODO dealing with the case where duplicates are forced due to parameters
        # TODO maybe only do this for a percentage of curricula, and randomly set the others OR instead of using [1:], use [1:__]
        # TODO test this
        for i in range(numberOfCurricula):
            curricula.append(bestCurriculum[1:])
            envId = np.random.randint(0, len(ENV_NAMES.ALL_ENVS))
            curricula[i].append(getEnvFromDifficulty(envId, envDifficulty))
        assert len(curricula) == numberOfCurricula
        return curricula
