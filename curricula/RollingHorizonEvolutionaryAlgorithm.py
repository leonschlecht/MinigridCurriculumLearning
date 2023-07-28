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
        self.useNSGA: bool = args.useNSGA
        self.crossoverProb = args.crossoverProb
        self.mutationProb = args.mutationProb
        self.crossoverEta = args.crossoverEta
        self.mutationEta = args.mutationEta

        # debug variable
        self.resX = None

    @staticmethod
    def getGenAndIdxOfBestIndividual(currentRewards):
        """
        Given the dict of currentRewards of the form {"gen1": [reward1, reward2, ...], "gen2": [reward1, reward2, ...], ... }
        Careful: if there is a "6.3e-2" instead of the full number in the string, then this will break
        return the generation Number (without the "gen" part) and the index in that list
        :param currentRewards:
        :return:
        """
        currentRewardList = np.array(list(currentRewards.values()))  # transform dict to list / matrix
        currentMaxRewardIdx = np.argmax(currentRewardList)  # highest idx in 1d list
        keyIndexPairOfMaxReward = np.unravel_index(currentMaxRewardIdx, currentRewardList.shape)
        genNrStr = list(currentRewards.keys())[keyIndexPairOfMaxReward[0]][len(GEN_PREFIX):]
        listIdx = int(keyIndexPairOfMaxReward[1])
        return genNrStr, listIdx

    def executeOneEpoch(self, epoch: int):
        sampling = IntegerRandomSampling()
        crossover = SBX(prob=self.crossoverProb, eta=self.crossoverEta, vtype=float, repair=RoundingRepair())
        mutation = PM(prob=self.mutationProb, eta=self.mutationEta, vtype=float, repair=RoundingRepair())
        if self.useNSGA:
            algorithm = NSGA2(pop_size=self.numCurric,
                              sampling=sampling,
                              crossover=crossover,
                              mutation=mutation,
                              eliminate_duplicates=True,
                              )
        else:
            algorithm = GA(pop_size=self.numCurric,
                           sampling=sampling,
                           crossover=crossover,
                           mutation=mutation,
                           eliminate_duplicates=True,
                           )

        # NSGA2 Default: # sampling: FloatRandomSampling = FloatRandomSampling(),
        # selection: TournamentSelection = TournamentSelection(func_comp=binary_tournament),
        # crossover: SBX = SBX(eta=15, prob=0.9),
        # mutation: PM = PM(eta=20),
        curricProblem = CurriculumProblem(self.curricula, self.objectives, self.inequalityConstr, self.xupper,
                                          self.paraEnvs, self)

        res = minimize(curricProblem,
                       algorithm,
                       termination=('n_gen', self.nGen),
                       seed=self.seed,
                       save_history=True,
                       verbose=False)
        self.resX = res.X
        # self.txtLogger.info(f"resX = {res.X} resF = {res.F}")

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
        This method is called from the curriculumProblem_eval method. It simulates one generation and returns the reward to pymoo
        :param evolX: the X parameter of the current RHEA population
        :param genNr: the number of the current generation
        :return: the rewards after the rolling horizon
        """
        curricula = self.evolXToCurriculum(evolX)
        self.curricula = curricula
        rewards = np.zeros(len(curricula))
        snapshotReward = np.zeros(len(curricula))
        genKey = GEN_PREFIX + str(genNr)
        for i in range(len(curricula)):
            rewardI = self.trainACurriculum(i, self.iterationsDone, genNr, curricula)
            snapshotReward[i] = rewardI[0]
            rewards[i] = np.sum(rewardI)
            # self.txtLogger.info(f"\n===curriculum {i} done===\n")
        self.currentRewardsDict[genKey] = rewards
        self.currentSnapshotRewards[genKey] = snapshotReward
        self.curriculaEnvDetails[genKey] = curricula
        # self.txtLogger.info("\n")
        #self.txtLogger.info(f"currentRewards after {genKey}: {self.currentRewardsDict}")
        #self.txtLogger.info(f"snapshot Rewards after {genKey}: {self.currentSnapshotRewards}")
        #self.txtLogger.info(f"currentEnvDetails for {genKey}: {self.curriculaEnvDetails[genKey]}")
        #self.txtLogger.info(f"\n\n===       Generation {genNr} done      ===\n")
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
