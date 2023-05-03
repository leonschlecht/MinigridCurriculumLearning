import argparse
import json
import os
from datetime import datetime
import numpy as np
import random
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize

import utils
from curricula.curriculumProblem import CurriculumProblem
from curricula import train, evaluate
from utils import ENV_NAMES
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX  # simulated binary crossover
from pymoo.operators.mutation.pm import PM  # polynomial mutation
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling


class RollingHorizonEvolutionaryAlgorithm:

    def __init__(self, txtLogger, startTime: datetime, args: argparse.Namespace, gamma=.9):
        assert args.envsPerCurric > 0
        assert args.numCurric > 0
        assert args.iterPerEnv > 0
        assert args.trainEpochs > 1
        self.args = args
        self.numCurric = args.numCurric
        self.envsPerCurric = args.envsPerCurric

        # Pymoo parameters
        objectives = 1
        xupper = len(ENV_NAMES.ALL_ENVS) - 1
        inequalityConstr = 0
        self.curricula = self.randomlyInitializeCurricula(args.numCurric, args.envsPerCurric)

        self.ITERATIONS_PER_ENV = args.iterPerEnv
        self.iterationsDone = 0
        self.txtLogger = txtLogger
        self.startTime = startTime
        self.selectedModel = utils.getEpochModelName(args.model, 0)  # TODO is this useful for 0?
        self.trainingEpochs = args.trainEpochs
        self.nGen = args.nGen
        self.trainEpochs = args.trainEpochs

        self.trainingInfoJson = {}
        self.logFilePath = os.getcwd() + "\\storage\\" + args.model + "\\status.json"  # TODO maybe outsource
        self.gamma = gamma
        self.currentRewards = {}
        self.curriculaEnvDetails = {}
        self.model = args.model

        self.txtLogger.info(f"curricula list start {self.curricula}")
        self.startTrainingLoop(objectives, inequalityConstr, xupper)

    def trainEachCurriculum(self, i: int, iterationsDone: int, genNr: int) -> int:
        """
        Simulates a horizon and returns the rewards obtained after evaluating the state at the end of the horizon
        """
        reward = 0
        # Save epochX -> epochX_curricI_genJ
        nameOfCurriculumI = utils.getModelWithCurricGenSuffix(self.selectedModel, i, GEN_PREFIX, genNr)
        # TODO this should not have the incorrect
        utils.copyAgent(src=self.selectedModel, dest=nameOfCurriculumI)
        for j in range(len(self.curricula[i])):
            iterationsDone = train.main(iterationsDone + self.ITERATIONS_PER_ENV, iterationsDone, nameOfCurriculumI,
                                        self.curricula[i][j], self.args, self.txtLogger)
            reward += ((self.gamma ** j) * evaluate.evaluateAgent(nameOfCurriculumI, self.args))  # TODO or (j+1) ?
            self.txtLogger.info(f"\tIterations Done {iterationsDone}")
            if j == 0:
                self.ITERATIONS_PER_ENV = iterationsDone + 1 # TODO test this ;
                utils.copyAgent(src=nameOfCurriculumI, dest=utils.getModelWithCandidatePrefix(
                    nameOfCurriculumI))  # save TEST_e1_curric0 -> + _CANDIDATE
            self.txtLogger.info(f"Trained iteration j={j} of curriculum {nameOfCurriculumI} ")
        self.txtLogger.info(f"Reward for curriculum {nameOfCurriculumI} = {reward}")
        return reward

    def printFinalLogs(self) -> None:
        """
        Prints the last logs, after the training is done
        """
        self.txtLogger.info("----TRAINING END-----")
        self.txtLogger.info(f"Best Curricula {self.trainingInfoJson[bestCurriculas]}")
        self.txtLogger.info(f"Trained in Envs {self.trainingInfoJson[selectedEnvs]}")
        self.txtLogger.info(f"Rewards: {self.trainingInfoJson[rewardsKey]}")

        now = datetime.now()
        timeDiff = (now - self.startTime).total_seconds()
        print(timeDiff)
        self.txtLogger.info(f"Time ended at {now} , total training time: {timeDiff}")
        self.txtLogger.info("-------------------\n\n")

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

    def initTrainingInfo(self) -> None:
        """
        Initializes the trainingInfo dictionary
        :return:
        """  # TODO maybe return sth
        self.trainingInfoJson = {selectedEnvs: [],
                                 bestCurriculas: [],
                                 curriculaEnvDetails: {},
                                 rewardsKey: {},
                                 actualPerformance: [],
                                 epochsDone: 1,
                                 startTime: self.startTime,
                                 numFrames: 0}
        self.saveJsonFile(self.logFilePath, self.trainingInfoJson)

    @staticmethod
    def saveJsonFile(path, jsonBody):
        with open(path, 'w') as f:
            f.write(json.dumps(jsonBody, indent=4, default=str))

    def updateTrainingInfo(self, epoch: int, bestCurriculum: list, currentRewards, currentScore: float, popX) -> None:
        self.trainingInfoJson[epochsDone] = epoch + 1
        self.trainingInfoJson[numFrames] = self.iterationsDone

        self.trainingInfoJson[selectedEnvs].append(bestCurriculum[0])
        self.trainingInfoJson[bestCurriculas].append(bestCurriculum)
        self.trainingInfoJson[rewardsKey] = currentRewards  # TODO test this ? or does this not overwrite everything
        self.trainingInfoJson[actualPerformance].append([currentScore, bestCurriculum])
        self.trainingInfoJson[curriculaEnvDetails]["epoch" + str(epoch)] = self.curriculaEnvDetails

        self.trainingInfoJson["currentListOfCurricula"] = self.curricula  # TODO is this useful?
        self.trainingInfoJson["curriculumListAsX"] = popX

        self.saveTrainingInfoToFile()
        # TODO how expensive is it to always overwrite everything?

    def logInfoAfterEpoch(self, epoch, currentBestCurriculum):
        """
        Logs relevant training info after a training epoch is done and the trainingInfo was updated
        :param epoch:
        :param currentBestCurriculum: the id of the current best curriculum
        :return:
        """
        selectedEnv = self.trainingInfoJson[selectedEnvs][-1]

        self.txtLogger.info(
            f"Best results in epoch {epoch} came from curriculum {currentBestCurriculum}")
        self.txtLogger.info(
            f"CurriculaEnvDetails {self.curriculaEnvDetails}; selectedEnv: {selectedEnv}")
        self.txtLogger.info(f"\nEPOCH: {epoch} SUCCESS\n")

    def startTrainingLoop(self, objectives: int, inequalityConstr, xupper):
        """
        Starts the training loop
        :param objectives: the no of objectives for pymoo
        :param inequalityConstr: ?
        :param xupper: the uppper bound for the evolutionary x variable
        :return: 
        """

        nsga = NSGA2(pop_size=len(self.curricula),
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

        for epoch in range(startEpoch, self.trainingEpochs):
            self.txtLogger.info(f"------------------------\nSTART EPOCH {epoch}\n----------------------")
            self.selectedModel = utils.getEpochModelName(self.model, epoch)
            curricProblem = CurriculumProblem(self.curricula, objectives, inequalityConstr, xupper, self)
            algorithm = GA(pop_size=len(self.curricula),
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
            self.iterationsDone += self.ITERATIONS_PER_ENV  # TODO use exact value
            nextModel = utils.getEpochModelName(self.model, epoch + 1)
            rewards["epoch" + str(epoch)] = self.currentRewards

            currentScore: float = np.max(list(self.currentRewards.values()))
            genOfBestIndividual, curricIdxOfBestIndividual = self.getGenAndIdxOfBestIndividual(self.currentRewards)
            currentBestModel = utils.getModelWithCurricGenSuffix(self.selectedModel, curricIdxOfBestIndividual,
                                                                 GEN_PREFIX, genOfBestIndividual)
            utils.copyAgent(src=currentBestModel, dest=nextModel)
            currentBestCurriculum = self.curriculaEnvDetails[GEN_PREFIX + genOfBestIndividual][
                curricIdxOfBestIndividual]

            # TODO-- what is res.X in this case? is it useless for next epoch ?
            self.updateTrainingInfo(epoch, currentBestCurriculum, rewards, currentScore, res.X)
            self.logInfoAfterEpoch(epoch, currentBestCurriculum)

            self.currentRewards = {}
            self.curriculaEnvDetails = {}

        self.printFinalLogs()
        print("final fitness:", res.F.sum())
        print("Final X = ", res.X)

    @staticmethod
    def randomlyInitializeCurricula(numberOfCurricula: int, envsPerCurriculum: int) -> list:
        """
        Initializes list of curricula randomly
        :param numberOfCurricula: how many curricula will be generated
        :param envsPerCurriculum: how many environment each curriculum has
        """
        curricula = []
        for i in range(numberOfCurricula):
            indices = random.sample(range(len(ENV_NAMES.ALL_ENVS)), envsPerCurriculum)
            newCurriculum = [ENV_NAMES.ALL_ENVS[idx] for idx in indices]
            curricula.append(newCurriculum)
        assert len(curricula) == numberOfCurricula
        return curricula

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
            self.startTime = datetime.fromisoformat(self.trainingInfoJson[startTime])
            # TODO calculate endTime during each save, and then claculate total training time
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
            self.txtLogger.info(f"Continung training from epoch {startEpoch}... ")
        else:
            self.txtLogger.info("Creating model. . .")
            train.main(0, 0, self.selectedModel, ENV_NAMES.DOORKEY_5x5, self.args, self.txtLogger)
            self.initTrainingInfo()
            startEpoch = 1
            utils.copyAgent(src=self.selectedModel,
                            dest=utils.getEpochModelName(self.model, startEpoch))  # copy epoch0 -> epoch1
            rewardsDict = {}
        return startEpoch, rewardsDict

    def saveTrainingInfoToFile(self):
        """
        Saves the training info into a local file called status.json in the main folder of the model's storage
        """
        with open(self.logFilePath, 'w') as f:
            f.write(json.dumps(self.trainingInfoJson, indent=4, default=str))

    def trainEveryCurriculum(self, evolX, genNr):
        """
        This method is called from the curriculumProblem_eval method
        :param evolX: the X parameter of the current RHEA population
        :param genNr: the number of the current generation
        :return: the rewards after the rolling horizon
        """
        curricula = self.evolXToCurriculum(evolX)
        self.curricula = curricula # TODO this is probably useless; maybe give it as param or store it otherwise. Im not sure how much other methods rely on this though
        rewards = np.zeros(len(curricula))
        for i in range(len(curricula)):
            rewardI = self.trainEachCurriculum(i, self.iterationsDone, genNr) # TODO. test if this actually uses correct curricula !!
            rewards[i] = rewardI
        self.currentRewards[GEN_PREFIX + str(genNr)] = rewards
        self.curriculaEnvDetails[GEN_PREFIX + str(genNr)] = curricula
        self.txtLogger.info(f"currentRewards for {genNr}: {self.currentRewards}")
        self.txtLogger.info(f"currentEnvDetails for {genNr}: {self.curriculaEnvDetails}")
        return np.array(rewards)

    @staticmethod
    def evolXToCurriculum(x):
        """
        Transforms the population.X to environment name string
        :param x:
        :return:
        """
        result = []

        for i in range(x.shape[0]):
            result.append([])
            curric = x[i]
            for j in range(curric.shape[0]):  # TODO bug if X is 1d ? / When is X 1d?
                result[i].append(ENV_NAMES.ALL_ENVS[x[i][j]])
        return result

    @staticmethod
    def createFirstGeneration(curriculaList):
        """
        Helper method that creates the biased first population of the RHEA
        Transform from environment language strings -> integers
        :return the transformed list containing integers representing the environment Nr
        """
        indices = []
        for i in range(len(curriculaList)):
            indices.append([])
            for env in curriculaList[i]:
                indices[i].append(ENV_NAMES.ALL_ENVS.index(env))
        return indices


def evaluateCurriculumResults(evaluationDictionary):
    # evaluationDictionary["actualPerformance"][0] ---> zeigt den avg reward des models zu jedem übernommenen Snapshot
    # evaluationDictionary["actualPerformance"][1] ---> zeigt die zuletzt benutzte Umgebung zu dem Zeitpunkt an
    #
    tmp = []
    i = 0
    for reward, env in tmp:
        print(reward, env)
        i += 1

    # Dann wollen wir sehen, wie das curriculum zu dem jeweiligen zeitpunkt ausgesehen hat.
    # # Aber warum? Und wie will man das nach 20+ durchläufen plotten


"""
foreach Iteration:
    Get current copy of the best RL Agent
    Intialize a Problem with Pymoo which will use copies of the best RL agent to evaluate solutions
        - The evaluate function receives an integer vector that represents a curriculum. Preferrably, we can use an integer encoding or
        evaluate will return the performance of the RL agent after performing the training with a given curriculum
        - the population consists of multiple curricula, which will all be tested
    we initialize an optimization algorithm
    We use the minimize function to run the optimization (minimize will call evaluate and update the population in between)
    we query the result of the minimize function to give us the best curriculum and use the timestep after the first phase of this curriculum as new best RL agent
"""

###### DEFINE CONSTANTS AND DICTIONARY KEYS #####

GEN_PREFIX = 'gen'

selectedEnvs = "selectedEnvs"
bestCurriculas = "bestCurriculas"
curriculaEnvDetails = "curriculaEnvDetails"
rewardsKey = "rewards"
actualPerformance = "actualPerformance"
epochsDone = "epochsDone"
startTime = "startTime"
numFrames = "numFrames"
