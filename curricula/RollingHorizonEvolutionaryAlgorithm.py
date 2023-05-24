import argparse
from datetime import datetime

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX  # simulated binary crossover
from pymoo.operators.mutation.pm import PM  # polynomial mutation
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize

import utils
from curricula.RollingHorizon import RollingHorizon
from curricula.curriculumProblem import CurriculumProblem
from utils import getEnvFromDifficulty, ENV_NAMES
from utils.curriculumHelper import *


class RollingHorizonEvolutionaryAlgorithm(RollingHorizon):

    def __init__(self, txtLogger, startTime: datetime, cmdLineString: str, args: argparse.Namespace):
        super().__init__(txtLogger, startTime, cmdLineString, args)
        self.nGen = args.nGen
        self.objectives = 1
        self.inequalityConstr = 0
        self.xupper = len(ENV_NAMES.ALL_ENVS) - 1

        # debug variable
        self.resX = None

    @staticmethod
    def getGenAndIdxOfBestIndividual(currentRewards):
        """
        Given the dict of currentRewards of the form {"gen1": [reward1, reward2, ...], "gen2": [reward1, reward2, ...], ... }
        return the generation Number (without the "gen" part) and the index in that list
        :param currentRewards:
        :return:
        """
        currentRewardList = np.array(list(currentRewards.values()))  # transform dict to list / matrix
        currentMaxRewardIdx = np.argmax(currentRewardList)  # highest idx in 1d list
        keyIndexPairOfMaxReward = np.unravel_index(currentMaxRewardIdx, currentRewardList.shape)
        return list(currentRewards.keys())[keyIndexPairOfMaxReward[0]][len(GEN_PREFIX):], \
            int(keyIndexPairOfMaxReward[1])

    def executeOneEpoch(self, epoch: int):
        nsga = NSGA2(pop_size=self.numCurric,
                     sampling=IntegerRandomSampling(),
                     crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                     mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                     eliminate_duplicates=True,
                     )
        # NSGA2 Default: # sampling: FloatRandomSampling = FloatRandomSampling(),
        # selection: TournamentSelection = TournamentSelection(func_comp=binary_tournament),
        # crossover: SBX = SBX(eta=15, prob=0.9),
        # mutation: PM = PM(eta=20),
        curricProblem = CurriculumProblem(self.curricula, self.objectives, self.inequalityConstr, self.xupper,
                                          self.paraEnvs, self)
        algorithm = GA(pop_size=self.numCurric,
                       sampling=IntegerRandomSampling(),
                       crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                       mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                       eliminate_duplicaets=True,
                       )
        res = minimize(curricProblem,
                       algorithm,
                       termination=('n_gen', self.nGen),
                       seed=self.seed,
                       save_history=True,
                       verbose=False)
        self.resX = res.X
        self.txtLogger.info(f"resX = {res.X} resF = {res.F}")

    def updateSpecificInfo(self, epoch):
        self.trainingInfoJson["resX"] = self.resX

    def getCurrentBestModel(self):
        genOfBestIndividual, curricIdxOfBestIndividual = self.getGenAndIdxOfBestIndividual(self.currentRewardsDict)
        currentBestModel = \
            utils.getModelWithCurricGenSuffix(self.selectedModel, curricIdxOfBestIndividual, genOfBestIndividual)
        return currentBestModel

    def getCurrentBestCurriculum(self) -> None:
        genOfBestIndividual, curricIdxOfBestIndividual = self.getGenAndIdxOfBestIndividual(self.currentRewardsDict)
        currentBestCurriculum = self.curriculaEnvDetails[GEN_PREFIX + genOfBestIndividual][
            curricIdxOfBestIndividual]
        return currentBestCurriculum

    def trainEveryCurriculum(self, evolX, genNr):
        """
        This method is called from the curriculumProblem_eval method
        :param evolX: the X parameter of the current RHEA population
        :param genNr: the number of the current generation
        :return: the rewards after the rolling horizon
        """
        curricula = self.evolXToCurriculum(evolX)
        self.curricula = curricula
        rewards = np.zeros(len(curricula))
        snapshotReward = np.zeros(len(curricula))
        for i in range(len(curricula)):
            rewardI = self.trainACurriculum(i, self.iterationsDone, genNr, curricula)
            snapshotReward[i] = rewardI[0]
            rewards[i] = np.sum(rewardI)
        self.currentRewardsDict[GEN_PREFIX + str(genNr)] = rewards
        self.currentSnapshotRewards[GEN_PREFIX + str(genNr)] = snapshotReward
        self.curriculaEnvDetails[GEN_PREFIX + str(genNr)] = curricula
        self.txtLogger.info(f"currentRewards for {genNr}: {self.currentRewardsDict}")
        self.txtLogger.info(f"snapshot Rewards for {genNr}: {self.currentSnapshotRewards}")
        self.txtLogger.info(f"currentEnvDetails for {genNr}: {self.curriculaEnvDetails}")
        return rewards

    def evolXToCurriculum(self, x):
        """
        Transforms the population.X to a list of environment name strings
        :param x:
        :return:
        """
        curriculumList = []
        for curriculumI in x:
            sublist = []
            for i in range(0, len(curriculumI), self.args.paraEnv):
                tmp = curriculumI[i:i + self.args.paraEnv]
                envNames = []
                for envId in tmp:
                    envNames.append(getEnvFromDifficulty(envId, self.envDifficulty))
                sublist.append(envNames)
            curriculumList.append(sublist)
        assert curriculumList != []
        assert len(curriculumList) == self.numCurric
        assert len(curriculumList[0]) == self.stepsPerCurric
        return curriculumList

    @staticmethod
    def createFirstGeneration(curriculaList):
        """
        Helper method that creates the biased first population of the RHEA
        Transform from environment language strings -> integers
        :return the transformed list containing integers representing the environment Nr
        """
        indices = []
        print("curricList", curriculaList)
        for i in range(len(curriculaList)):
            indices.append([])
            for env in curriculaList[i]:
                indices[i].append(ENV_NAMES.ALL_ENVS.index(env))
        return indices

    def getCurriculumName(self, i, genNr):
        assert genNr >= 0, "genNr must be a positive number in RHEA"
        return utils.getModelWithCurricGenSuffix(self.selectedModel, i, genNr)
