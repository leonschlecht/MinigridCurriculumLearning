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
from scripts import train, evaluate
from utils import ENV_NAMES, getModelWithCandidatePrefix
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
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
        self.selectedModel = utils.getEpochModelName(args.model, 0)  # TODO is this useful ?
        self.trainingEpochs = args.trainEpochs
        self.nGen = args.nGen
        self.trainEpochs = args.trainEpochs

        self.trainingInfoJson = {}
        self.logFilePath = os.getcwd() + "\\storage\\" + args.model + "\\status.json"  # TODO maybe outsource
        self.gamma = gamma
        self.currentRewards = {}
        self.model = args.model

        self.txtLogger.info(f"curricula list start {self.curricula}")
        self.startTrainingLoop(objectives, inequalityConstr, xupper)

    def trainEachCurriculum(self, i: int, iterationsDone: int, genNr: int) -> int:
        """
        Simulates a horizon and returns the rewards obtained after evaluating the state at the end of the horizon
        """
        reward = 0
        # Save epochX -> epochX_curricI_genJ
        nameOfCurriculumI = utils.getModelWithCurricGenSuffix(self.selectedModel, i, genNr)
        utils.copyAgent(src=self.selectedModel, dest=nameOfCurriculumI)
        for j in range(len(self.curricula[i])):
            iterationsDone = train.main(iterationsDone + self.ITERATIONS_PER_ENV, nameOfCurriculumI,
                                        self.curricula[i][j], self.args, self.txtLogger)
            reward += ((self.gamma ** j) * evaluate.evaluateAgent(nameOfCurriculumI, self.args))  # TODO or (j+1) ?
            self.txtLogger.info(f"\tIterations Done {iterationsDone}")
            if j == 0:
                utils.copyAgent(src=nameOfCurriculumI, dest=utils.getModelWithCandidatePrefix(
                    nameOfCurriculumI))  # save TEST_e1_curric0 -> + _CANDIDATE
            self.txtLogger.info(f"Trained iteration j={j} of curriculum {nameOfCurriculumI} ")
        self.txtLogger.info(f"Reward for curriculum {nameOfCurriculumI} = {reward}")
        return reward

    def getCurriculaEnvDetails(self) -> dict:
        """
        Returns a dictionary with the environments of each curriculum
        { 0: [env1, env2, ], 1: [env1, env2, ], ... }
        """
        fullEnvList = {}
        for i in range(len(self.curricula)):
            fullEnvList[i] = self.curricula[i]
        return fullEnvList

    def printFinalLogs(self) -> None:
        """
        Prints the last logs, after the training is done
        """
        self.txtLogger.info("----TRAINING END-----")
        self.txtLogger.info(f"Best Curricula {self.trainingInfoJson['bestCurriculaIds']}")
        self.txtLogger.info(f"Trained in Envs {self.trainingInfoJson['selectedEnvs']}")
        self.txtLogger.info(f"Rewards: {self.trainingInfoJson['rewards']}")

        now = datetime.now()
        timeDiff = (now - self.startTime).total_seconds()
        print(timeDiff)
        self.txtLogger.info(f"Time ended at {now} , total training time: {timeDiff}")
        self.txtLogger.info("-------------------\n\n")

    @staticmethod
    def getGenAndIdxOfBestIndividual(currentRewards):
        currentRewardList = np.array(list(currentRewards.values()))  # transform dict to list / matrix
        currentMaxRewardIdx = np.argmax(currentRewardList)  # highest idx in 1d list
        keyIndexPairOfMaxReward = np.unravel_index(currentMaxRewardIdx, currentRewardList.shape)
        return list(currentRewards.keys())[keyIndexPairOfMaxReward[0]][3:], \
            int(keyIndexPairOfMaxReward[1]) # TODO len('gen') vs 3

    def initTrainingInfo(self):
        self.trainingInfoJson = {"selectedEnvs": [],
                                 "bestCurriculaIds": [],
                                 "curriculaEnvDetails": {},
                                 "rewards": {},
                                 "actualPerformance": [],
                                 "epochsDone": 1,
                                 "startTime": self.startTime,
                                 "numFrames": 0}
        self.saveJsonFile(self.logFilePath, self.trainingInfoJson)

    @staticmethod
    def saveJsonFile(path, jsonBody):
        with open(path, 'w') as f:
            f.write(json.dumps(jsonBody, indent=4, default=str))

    def updateTrainingInfo(self, epoch, idOfBestCurriculum, rewards, currentScore, popX) -> None:
        self.trainingInfoJson["epochsDone"] = epoch + 1
        self.trainingInfoJson["numFrames"] = self.iterationsDone

        bestCurriculum = self.curricula[idOfBestCurriculum]
        self.trainingInfoJson["selectedEnvs"].append(bestCurriculum[0])
        self.trainingInfoJson["bestCurriculaIds"].append(idOfBestCurriculum)
        self.trainingInfoJson["rewards"] = rewards
        self.trainingInfoJson["actualPerformance"].append([currentScore, bestCurriculum])
        envDetailsOfCurrentEpoch = self.getCurriculaEnvDetails()
        self.trainingInfoJson["curriculaEnvDetails"]["epoch" + str(epoch)] = envDetailsOfCurrentEpoch
        self.trainingInfoJson["curriculaEnvDetails"]["epoch" + str(epoch + 1)] = self.curricula  # save as backup

        self.trainingInfoJson["currentListOfCurricula"] = self.curricula
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
        selectedEnv = self.trainingInfoJson["selectedEnvs"][-1]

        self.txtLogger.info(
            f"Best results in epoch {epoch} came from curriculum {self.curricula[currentBestCurriculum]}")
        self.txtLogger.info(
            f"CurriculaEnvDetails {self.getCurriculaEnvDetails()}; selectedEnv: {selectedEnv}")
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

        # TODO fix curriculaEnvDetails
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
            # TODO maybe set biased pop for 1st epoch 1st generation
            if epoch == 1:
                X = self.createFirstGeneration(self.curricula)
                pop = Population.new("X", X)
            res = minimize(curricProblem,
                           algorithm,
                           termination=('n_gen', self.nGen),
                           seed=1,
                           save_history=True,
                           verbose=False)
            self.txtLogger.info(f"resX = {res.X} resF = {res.F}")
            self.iterationsDone += self.ITERATIONS_PER_ENV  # TODO use exact value
            nextModel = utils.getEpochModelName(self.model, epoch + 1)
            currentBestCurriculum = np.argmax(self.currentRewards)
            rewards["epoch" + str(epoch)] = self.currentRewards

            currentScore = max(list(self.currentRewards.values()))
            genOfBestIndividual, curricIdxOfBestIndividual = self.getGenAndIdxOfBestIndividual(self.currentRewards)
            currentBestModel = utils.getModelWithCurricGenSuffix(self.selectedModel, curricIdxOfBestIndividual,
                                                                 genOfBestIndividual)
            utils.copyAgent(src=currentBestModel, dest=nextModel)

            # TODO assert that res.X == curric[bestScore] ; but evolXToCurric needs to handle 1d then ; or what if res.X has multiple?
            self.updateTrainingInfo(epoch, currentBestCurriculum, rewards, currentScore, res.X)
            self.logInfoAfterEpoch(epoch, currentBestCurriculum)

        self.printFinalLogs()
        print("final fitness:", res.F.sum())
        print("Final X = ", res.X)
        print("#curricula:", len(self.curricula))

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

            self.iterationsDone = self.trainingInfoJson["numFrames"]
            startEpoch = self.trainingInfoJson["epochsDone"]
            rewards = self.trainingInfoJson["rewards"]
            self.startTime = datetime.fromisoformat(self.trainingInfoJson["startTime"])  # TODO this is useless
            # self.trainEpochs = # TODO load to prevent accidental overwrite ?
            # delete existing folders, that were created ---> maybe just last one because others should be finished ...
            for k in range(self.numCurric):
                path = utils.getModelWithCurricSuffix(self.model, startEpoch, k)
                if utils.deleteModelIfExists(path):
                    print("deleted", k)
                    snapshotPath = utils.getModelWithCandidatePrefix(path)
                    utils.deleteModelIfExists(snapshotPath)
                else:
                    print("Nothing to delete", k)
                    break
            # assert len(self.curricula) == self.trainingInfoJson["curriculaEnvDetails"]["epoch0"] # TODO fix
            # self.curricula = self.trainingInfoJson["currentListOfCurricula"] # TODO fix
            self.txtLogger.info(f"Continung training from epoch {startEpoch}... ")
        else:
            self.txtLogger.info("Creating model. . .")
            self.iterationsDone = train.main(0, self.selectedModel, ENV_NAMES.DOORKEY_5x5, self.args, self.txtLogger)
            self.initTrainingInfo()
            utils.copyAgent(src=self.selectedModel, dest=utils.getEpochModelName(self.model, 1))  # copy e0 -> e1
            startEpoch = 1
            rewards = {}
        return startEpoch, rewards

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
        rewards = []  # TODO replcae list with np array directly
        for i in range(len(curricula)):
            rewardI = self.trainEachCurriculum(i, self.iterationsDone, genNr)
            rewards.append(rewardI)
        self.currentRewards["gen" + str(genNr)] = rewards
        self.curricula = curricula
        self.txtLogger.info(f"currentRewards for {genNr}: {self.currentRewards}")
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
            for j in range(curric.shape[0]):  # TODO bug if X is 1d ?
                rounded = round(x[i][j])  # TODO round is not necessary; it should only assert it is INT7
                result[i].append(ENV_NAMES.ALL_ENVS[rounded])
        return result

    @staticmethod
    def createFirstGeneration(curriculaList):
        """
        Helper method that creates the biased first population of the RHEA
        Transform from environment language string -> numbers
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
