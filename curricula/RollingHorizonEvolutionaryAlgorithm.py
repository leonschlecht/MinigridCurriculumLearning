import argparse
import os
import numpy as np
from numpy import ndarray
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
        assert 0 < args.paraEnv <= len(ENV_NAMES.ALL_ENVS)

        self.args = args
        self.numCurric = args.numCurric
        self.envsPerCurric = args.envsPerCurric
        self.cmdLineString = cmdLineString
        self.lastEpochStartTime = startTime
        self.envDifficulty = 0
        self.exactIterationsSet = False
        self.seed = args.seed
        self.paraEnvs = args.paraEnv  # the amount of envs that are trained in parallel

        # Pymoo parameters
        objectives = 1
        xupper = len(ENV_NAMES.ALL_ENVS) - 1
        inequalityConstr = 0
        self.curricula = randomlyInitializeCurricula(args.numCurric, args.envsPerCurric, self.envDifficulty)

        self.ITERATIONS_PER_ENV = args.iterPerEnv
        self.iterationsDone = 0
        self.txtLogger = txtLogger
        self.selectedModel = utils.getEpochModelName(args.model, 0)
        self.totalEpochs = args.trainEpochs
        self.nGen = args.nGen
        self.trainingTime = 0

        self.maxReward = calculateMaxReward(1, gamma)  # TODO maybe remove this because it became useless
        print("maxReward", self.maxReward)

        self.trainingInfoJson = {}
        self.logFilePath = os.getcwd() + "\\storage\\" + args.model + "\\status.json"  # TODO maybe outsource
        self.gamma = gamma
        self.currentRewards = {}
        self.currentSnapshotRewards = {}
        self.curriculaEnvDetails = {}
        self.model = args.model
        self.txtLogger.info(f"curricula list start {self.curricula}")
        self.startTrainingLoop(objectives, inequalityConstr, xupper)

    def trainEachCurriculum(self, i: int, iterationsDone: int, genNr: int, curricula) -> ndarray:
        """
        Simulates a horizon and returns the rewards obtained after evaluating the state at the end of the horizon
        """
        reward = np.zeros(len(curricula[i]))
        # Save epochX -> epochX_curricI_genJ
        nameOfCurriculumI = utils.getModelWithCurricGenSuffix(self.selectedModel, i, GEN_PREFIX, genNr)
        utils.copyAgent(src=self.selectedModel, dest=nameOfCurriculumI)
        initialIterationsDone = iterationsDone
        for j in range(len(curricula[i])):
            print("\t curricula[i][j] = ", curricula[i][j])
            iterationsDone = train.startTraining(iterationsDone + self.ITERATIONS_PER_ENV, iterationsDone,
                                                 nameOfCurriculumI, curricula[i][j], self.args, self.txtLogger)
            reward[j] = ((self.gamma ** j) * evaluate.evaluateAgent(nameOfCurriculumI, self.envDifficulty, self.args))
            self.txtLogger.info(f"\tIterations Done {iterationsDone}")
            if j == 0:
                self.saveFirstStepOfModel(iterationsDone - initialIterationsDone, nameOfCurriculumI)  # TODO testfor ep0
            self.txtLogger.info(f"\tTrained iteration j={j} of curriculum {nameOfCurriculumI}")
            self.txtLogger.info(f"\tReward for curriculum {nameOfCurriculumI} = {reward} (1 entry = 1 env)\n\n")
        return reward

    def saveFirstStepOfModel(self, exactIterationsPerEnv: int, nameOfCurriculumI: str):
        if not self.exactIterationsSet:  # TODO refactor this to common method
            self.exactIterationsSet = True
            self.ITERATIONS_PER_ENV = exactIterationsPerEnv - 1
        utils.copyAgent(src=nameOfCurriculumI, dest=utils.getModelWithCandidatePrefix(
            nameOfCurriculumI))  # save TEST_e1_curric0 -> + _CANDIDATE
        self.txtLogger.info(f"ITERATIONS PER ENV = {self.ITERATIONS_PER_ENV}")

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
        # NSGA2 Default: # sampling: FloatRandomSampling = FloatRandomSampling(),
        # selection: TournamentSelection = TournamentSelection(func_comp=binary_tournament),
        # crossover: SBX = SBX(eta=15, prob=0.9),
        # mutation: PM = PM(eta=20),
        startEpoch, rewards = self.initializeTrainingVariables(os.path.exists(self.logFilePath))
        for epoch in range(startEpoch, self.totalEpochs):
            self.txtLogger.info(
                f"\n--------------------------------------------------------------\n                     START EPOCH {epoch}\n--------------------------------------------------------------\n")
            self.selectedModel = utils.getEpochModelName(self.model, epoch)
            curricProblem = CurriculumProblem(self.curricula, objectives, inequalityConstr, xupper, self.paraEnvs, self)
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
            self.txtLogger.info(f"resX = {res.X} resF = {res.F}")
            self.iterationsDone += self.ITERATIONS_PER_ENV
            nextModel = utils.getEpochModelName(self.model, epoch + 1)
            rewards["epoch" + str(epoch)] = self.currentRewards  # TODO also for snapshot rewards?

            bestCurriculumScore: float = np.max(list(self.currentRewards.values()))
            currentSnapshotScore: float = np.max(list(self.currentSnapshotRewards.values()))
            genOfBestIndividual, curricIdxOfBestIndividual = self.getGenAndIdxOfBestIndividual(self.currentRewards)
            currentBestModel = utils.getModelWithCurricGenSuffix(self.selectedModel, curricIdxOfBestIndividual,
                                                                 GEN_PREFIX, genOfBestIndividual)
            utils.copyAgent(src=currentBestModel, dest=nextModel)
            currentBestCurriculum = self.curriculaEnvDetails[GEN_PREFIX + genOfBestIndividual][
                curricIdxOfBestIndividual]
            self.envDifficulty = calculateEnvDifficulty(currentSnapshotScore, self.maxReward)

            updateTrainingInfo(self.trainingInfoJson, epoch, currentBestCurriculum, rewards, bestCurriculumScore,
                               currentSnapshotScore, self.iterationsDone, self.envDifficulty, self.lastEpochStartTime,
                               self.curricula, self.curriculaEnvDetails, self.logFilePath, res.X)
            logInfoAfterEpoch(epoch, currentBestCurriculum, bestCurriculumScore, self.trainingInfoJson, self.txtLogger,
                              self.maxReward, self.totalEpochs)

            self.currentRewards = {}
            self.currentSnapshotRewards = {}
            self.curriculaEnvDetails = {}
            self.lastEpochStartTime = datetime.now()

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
            # TODO maybe do the deletion automatically, but it doesnt matter
            """
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
            """
            self.txtLogger.info(f"Continung training from epoch {startEpoch}... [total epochs: {self.totalEpochs}]")
        else:
            self.txtLogger.info("Creating model. . .")
            train.startTraining(0, 0, self.selectedModel, [getEnvFromDifficulty(0, self.envDifficulty)], self.args,
                                self.txtLogger)
            self.trainingInfoJson = initTrainingInfo(self.cmdLineString, self.logFilePath, self.seed, self.args)
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
        snapshotReward = np.zeros(len(curricula))
        for i in range(len(curricula)):
            rewardI = self.trainEachCurriculum(i, self.iterationsDone, genNr, curricula)
            snapshotReward[i] = rewardI[0]
            rewards[i] = np.sum(rewardI)
        self.currentRewards[GEN_PREFIX + str(genNr)] = rewards
        self.currentSnapshotRewards[GEN_PREFIX + str(genNr)] = snapshotReward
        self.curriculaEnvDetails[GEN_PREFIX + str(genNr)] = curricula
        self.txtLogger.info(f"currentRewards for {genNr}: {self.currentRewards}")
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
        assert len(curriculumList[0]) == self.envsPerCurric
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
