import argparse
import os
import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize

import utils
from curricula.curriculumProblem import CurriculumProblem
from curricula import train, evaluate
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX  # simulated binary crossover
from pymoo.operators.mutation.pm import PM  # polynomial mutation
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling

from utils.curriculumHelper import *


class RollingHorizonEvolutionaryAlgorithm:

    def __init__(self, txtLogger, startTime: datetime, cmdLineString: str, args: argparse.Namespace, gamma=.9):
        assert args.envsPerCurric > 0
        assert args.numCurric > 0
        assert args.iterPerEnv > 0
        assert args.trainEpochs > 1
        self.args = args
        self.numCurric = args.numCurric
        self.envsPerCurric = args.envsPerCurric
        self.cmdLineString = cmdLineString
        self.lastEpochStartTime = startTime
        self.envDifficulty = 0
        self.exactIterationsSet = False

        # Pymoo parameters
        objectives = 1
        xupper = len(ENV_NAMES.ALL_ENVS) - 1
        inequalityConstr = 0
        self.curricula = randomlyInitializeCurricula(args.numCurric, args.envsPerCurric, self.envDifficulty)

        self.ITERATIONS_PER_ENV = args.iterPerEnv
        self.iterationsDone = 0
        self.txtLogger = txtLogger
        self.selectedModel = utils.getEpochModelName(args.model, 0)  # TODO is this useful for 0?
        self.totalEpochs = args.trainEpochs
        self.nGen = args.nGen
        self.trainingTime = 0

        MAX_REWARD_PER_ENV = 1
        maxReward = 0
        for j in range(args.numCurric):
            maxReward += ((gamma ** j) * MAX_REWARD_PER_ENV * args.numCurric)
        self.maxReward = maxReward
        print("maxReward", self.maxReward)

        self.trainingInfoJson = {}
        self.logFilePath = os.getcwd() + "\\storage\\" + args.model + "\\status.json"  # TODO maybe outsource
        self.gamma = gamma
        self.currentRewards = {}
        self.curriculaEnvDetails = {}
        self.model = args.model
        self.txtLogger.info(f"curricula list start {self.curricula}")
        self.startTrainingLoop(objectives, inequalityConstr, xupper)

    def trainEachCurriculum(self, i: int, iterationsDone: int, genNr: int, curricula) -> int:
        """
        Simulates a horizon and returns the rewards obtained after evaluating the state at the end of the horizon
        """
        reward = 0
        # Save epochX -> epochX_curricI_genJ
        nameOfCurriculumI = utils.getModelWithCurricGenSuffix(self.selectedModel, i, GEN_PREFIX, genNr)
        utils.copyAgent(src=self.selectedModel, dest=nameOfCurriculumI)
        for j in range(len(curricula[i])):
            iterationsDone = train.main(iterationsDone + self.ITERATIONS_PER_ENV, iterationsDone, nameOfCurriculumI,
                                        curricula[i][j], self.args, self.txtLogger)
            reward += ((self.gamma ** j) * evaluate.evaluateAgent(nameOfCurriculumI, self.envDifficulty,
                                                                  self.args))  # TODO or (j+1) ?

            self.txtLogger.info(f"\tIterations Done {iterationsDone}")
            if j == 0:
                if not self.exactIterationsSet:
                    self.exactIterationsSet = True
                    self.ITERATIONS_PER_ENV = iterationsDone - 1  # TODO test this ; maybe you can remove the -1 (or add it)
                    self.txtLogger.info("first iteration set; Iter_per_env ...!")  # TODO remove
                utils.copyAgent(src=nameOfCurriculumI, dest=utils.getModelWithCandidatePrefix(
                    nameOfCurriculumI))  # save TEST_e1_curric0 -> + _CANDIDATE
            self.txtLogger.info(f"\tTrained iteration j={j} of curriculum {nameOfCurriculumI}\n")
        self.txtLogger.info(f"Reward for curriculum {nameOfCurriculumI} = {reward}\n\n")
        return reward

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

    def updateTrainingInfo(self, epoch: int, bestCurriculum: list, currentRewards, currentScore: float, popX) -> None:
        """
        Updates the training info dictionary
        :param epoch: current epoch
        :param bestCurriculum: the curriculum that had the highest reward in the latest epoch
        :param currentRewards: the dict of rewards for each generation and each curriculum
        :param currentScore: the current best score
        :param popX: the pymoo X parameter for debugging purposes
        """
        self.trainingInfoJson[epochsDone] = epoch + 1
        self.trainingInfoJson[numFrames] = self.iterationsDone

        self.trainingInfoJson[selectedEnvs].append(bestCurriculum[0])
        self.trainingInfoJson[bestCurriculas].append(bestCurriculum)
        self.trainingInfoJson[rewardsKey] = currentRewards  # TODO test this ? or does this not overwrite everything
        self.trainingInfoJson[actualPerformance].append([currentScore, bestCurriculum])
        self.trainingInfoJson[curriculaEnvDetails]["epoch" + str(epoch)] = self.curriculaEnvDetails
        self.trainingInfoJson[difficultyKey].append(self.envDifficulty)

        now = datetime.now()
        timeSinceLastEpoch = (now - self.lastEpochStartTime).total_seconds()
        self.trainingInfoJson[epochTrainingTime].append(timeSinceLastEpoch)
        self.trainingInfoJson[sumTrainingTime] += timeSinceLastEpoch
        self.lastEpochStartTime = now

        # Debug Logs
        self.trainingInfoJson["currentListOfCurricula"] = self.curricula  # TODO is this useful?
        self.trainingInfoJson["curriculumListAsX"] = popX

        saveTrainingInfoToFile(self.logFilePath, self.trainingInfoJson)
        # TODO how expensive is it to always overwrite everything?

    def startTrainingLoop(self, objectives: int, inequalityConstr, xupper):
        """
        Starts the training loop
        :param objectives: the no of objectives for pymoo
        :param inequalityConstr: ?
        :param xupper: the uppper bound for the evolutionary x variable
        :return: 
        """
        nsga = NSGA2(pop_size=self.numCurric,
                     sampling=IntegerRandomSampling(),
                     crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                     mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                     eliminate_duplicates=True,
                     )
        # TODO nsga result.X has potentially multiple entries ?
        # NSGA Default:
        # sampling: FloatRandomSampling = FloatRandomSampling(),
        # selection: TournamentSelection = TournamentSelection(func_comp=binary_tournament),
        # crossover: SBX = SBX(eta=15, prob=0.9),
        # mutation: PM = PM(eta=20),
        startEpoch, rewards = self.initializeTrainingVariables(os.path.exists(self.logFilePath))
        for epoch in range(startEpoch, self.totalEpochs):
            self.txtLogger.info(
                f"\n--------------------------------------------------------------\n                     START EPOCH {epoch}\n--------------------------------------------------------------\n")
            self.selectedModel = utils.getEpochModelName(self.model, epoch)
            curricProblem = CurriculumProblem(self.curricula, objectives, inequalityConstr, xupper, self)
            algorithm = GA(pop_size=self.numCurric,
                           sampling=IntegerRandomSampling(),
                           crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                           mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                           eliminate_duplicaets=True,
                           )
            res = minimize(curricProblem,
                           algorithm,
                           termination=('n_gen', self.nGen),
                           seed=1,
                           save_history=True,
                           verbose=False)
            self.txtLogger.info(f"resX = {res.X} resF = {res.F}")
            self.iterationsDone += self.ITERATIONS_PER_ENV
            nextModel = utils.getEpochModelName(self.model, epoch + 1)
            rewards["epoch" + str(epoch)] = self.currentRewards

            currentScore: float = np.max(list(self.currentRewards.values()))
            genOfBestIndividual, curricIdxOfBestIndividual = self.getGenAndIdxOfBestIndividual(self.currentRewards)
            currentBestModel = utils.getModelWithCurricGenSuffix(self.selectedModel, curricIdxOfBestIndividual,
                                                                 GEN_PREFIX, genOfBestIndividual)
            utils.copyAgent(src=currentBestModel, dest=nextModel)
            currentBestCurriculum = self.curriculaEnvDetails[GEN_PREFIX + genOfBestIndividual][
                curricIdxOfBestIndividual]

            self.updateTrainingInfo(epoch, currentBestCurriculum, rewards, currentScore, res.X)
            logInfoAfterEpoch(epoch, currentBestCurriculum, currentScore, self.trainingInfoJson, self.txtLogger, self.maxReward, self.totalEpochs)

            self.currentRewards = {}
            self.curriculaEnvDetails = {}
            self.envDifficulty = calculateEnvDifficulty(currentScore, self.maxReward)

        printFinalLogs(self.trainingInfoJson, self.txtLogger)
        print("final fitness:", res.F.sum())
        print("Final X = ", res.X)


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
            rewardsDict = self.trainingInfoJson[rewardsKey]

            # delete existing folders, that were created ---> maybe just last one because others should be finished ...
            for k in range(self.numCurric):
                path = utils.getModelWithCurricSuffix(self.model, startEpoch, k)
                if utils.deleteModelIfExists(path):
                    print("deleted", k)
                    snapshotPath = utils.getModelWithCandidatePrefix(path)
                    utils.deleteModelIfExists(snapshotPath)
                    # TODO test if delete _gen folders; OR probably get prefix -> look for them in list, delete all of these folders that contain it
                else:
                    self.txtLogger.info(f"Nothing to delete {k}")
                    break
            self.txtLogger.info(f"Continung training from epoch {startEpoch}... [total epochs: {self.totalEpochs}]")
        else:
            self.txtLogger.info("Creating model. . .")
            train.main(0, 0, self.selectedModel, getEnvFromDifficulty(0, self.envDifficulty), self.args, self.txtLogger)
            self.trainingInfoJson = initTrainingInfo(self.cmdLineString, self.logFilePath)
            startEpoch = 1
            utils.copyAgent(src=self.selectedModel,
                            dest=utils.getEpochModelName(self.model, startEpoch))  # copy epoch0 -> epoch1
            self.txtLogger.info(f"\nThe training will go on for {self.totalEpochs} epochs\n")
            rewardsDict = {}
        return startEpoch, rewardsDict

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
        for i in range(len(curricula)):
            rewardI = self.trainEachCurriculum(i, self.iterationsDone, genNr, curricula)
            rewards[i] = rewardI
        self.currentRewards[GEN_PREFIX + str(genNr)] = rewards
        self.curriculaEnvDetails[GEN_PREFIX + str(genNr)] = curricula
        self.txtLogger.info(f"currentRewards for {genNr}: {self.currentRewards}")
        self.txtLogger.info(f"currentEnvDetails for {genNr}: {self.curriculaEnvDetails}")
        return rewards

    def evolXToCurriculum(self, x):
        """
        Transforms the population.X to a list of environment name strings
        :param x:
        :return:
        """
        result = []

        for i in range(x.shape[0]):
            result.append([])
            curric = x[i]
            for j in range(curric.shape[0]):  # TODO bug if X is 1d ? / When is X 1d?
                result[i].append(getEnvFromDifficulty(x[i][j], self.envDifficulty))
        return result

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
