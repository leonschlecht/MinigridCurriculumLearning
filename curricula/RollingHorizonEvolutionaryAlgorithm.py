import json
import os
from datetime import datetime
import numpy as np
import random
from pymoo.algorithms.soo.nonconvex.ga import GA

import utils
from curricula.curriculumProblem import CurriculumProblem
from scripts import train, evaluate
from utils import ENV_NAMES, getModelWithCandidatePrefix
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
from pymoo.operators.crossover.sbx import SBX  # simulated binary crossover
from pymoo.operators.mutation.pm import PM  # polynomial mutation
from pymoo.operators.repair.rounding import \
    RoundingRepair  # falls ein Operator kommazahlen ausgibt, werden diese gerundet
from pymoo.operators.sampling.rnd import IntegerRandomSampling  # Sampling neuer Individuuen als integers


class RollingHorizonEvolutionaryAlgorithm:

    def __init__(self, txtLogger, startTime: datetime, args, gamma=.9):
        assert args.envsPerCurriculum > 0
        assert args.numberOfCurricula > 0
        assert args.iterationsPerEnv > 0
        assert args.curriculumEpochs > 1

        # Pymoo parameters
        objectives = 1
        # curric1 = [ENV_NAMES.DOORKEY_5x5, ENV_NAMES.DOORKEY_5x5, ENV_NAMES.DOORKEY_16x16, ENV_NAMES.DOORKEY_6x6]
        # curric2 = [ENV_NAMES.DOORKEY_8x8, ENV_NAMES.DOORKEY_16x16, ENV_NAMES.DOORKEY_16x16, ENV_NAMES.DOORKEY_16x16]
        xupper = len(ENV_NAMES.ALL_ENVS) - 1
        self.curricula = self.randomlyInitializeCurricula(args.numberOfCurricula, args.envsPerCurriculum)
        inequalityConstr = 0

        self.ITERATIONS_PER_ENV = args.iterationsPerEnv
        self.iterationsDone = 0
        self.txtLogger = txtLogger
        self.args = args
        self.startTime = startTime
        self.selectedModel = args.model + "\\epoch_0"
        self.nGen = args.curriculumEpochs

        self.trainingInfoJson = {}
        self.logFilePath = os.getcwd() + "\\storage\\" + args.model + "\\status.json"
        self.gamma = gamma
        self.currentRewards = []

        self.startTrainingLoop(objectives, inequalityConstr, xupper)

    def trainEachCurriculum(self, i: int, iterationsDone: int) -> int:
        """
        Simulates a horizon and returns the rewards obtained after evaluating the state at the end of the horizon
        """
        reward = 0
        nameOfCurriculumI = utils.getModelName(self.selectedModel, i)  # Save TEST_e1 --> TEST_e1_curric0
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

    def initializeRewards(self):
        """
        Loads the rewards given the state from previous training, or empty initializes them if first time run
        This assumes that the curricula are numbered from 1 to N (as opposed to using their names or something similar)
        """
        assert len(self.curricula) > 0
        rewards = {}
        for i in range(len(self.curricula)):
            rewards[str(i)] = []
        return rewards

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

    def initTrainingInfo(self, rewards, pretrainingIterations):
        self.trainingInfoJson = {"selectedEnvs": [],
                                 "bestCurriculaIds": [],
                                 "curriculaEnvDetails": {},
                                 "rewards": rewards,
                                 "actualPerformance": [],
                                 "epochsDone": 1,
                                 "startTime": self.startTime,
                                 "numFrames": pretrainingIterations}
        self.saveJsonFile(self.logFilePath, self.trainingInfoJson)

    @staticmethod
    def saveJsonFile(path, jsonBody):
        with open(path, 'w') as f:
            f.write(json.dumps(jsonBody, indent=4, default=str))

    def updateTrainingInfo(self, epoch, currentBestCurriculum, rewards, currentScore, popX) -> None:
        self.trainingInfoJson["epochsDone"] = epoch + 1
        self.trainingInfoJson["numFrames"] = self.iterationsDone

        selectedEnv = self.curricula[currentBestCurriculum][0]
        self.trainingInfoJson["selectedEnvs"].append(selectedEnv)
        self.trainingInfoJson["bestCurriculaIds"].append(currentBestCurriculum)
        self.trainingInfoJson["rewards"] = rewards
        self.trainingInfoJson["actualPerformance"].append([currentScore, selectedEnv])
        envDetailsOfCurrentEpoch = self.getCurriculaEnvDetails()
        self.trainingInfoJson["curriculaEnvDetails"]["epoch" + str(epoch)] = envDetailsOfCurrentEpoch
        self.trainingInfoJson["curriculaEnvDetails"]["epoch" + str(epoch + 1)] = self.curricula  # save as backup

        self.trainingInfoJson["currentCurriculumList"] = self.evolXToCurriculum(popX)
        self.trainingInfoJson["curriculumListAsX"] = popX

        self.saveTrainingInfoToFile()
        # TODO how expensive is it to always overwrite everything?

    def logRelevantInfo(self, epoch, currentBestCurriculum):
        """
        Logs relevant training info after a training epoch is done and the trainingInfo was updated
        :param epoch:
        :param currentBestCurriculum: the id of the current best curriculum
        :return:
        """
        selectedEnv = self.trainingInfoJson["selectedEnvs"][-1]

        self.txtLogger.info(f"Best results in epoch {epoch} came from curriculum {currentBestCurriculum}")
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
        curricProblem = CurriculumProblem(self.curricula, objectives, inequalityConstr, xupper, self)

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

        ga = GA(pop_size=len(self.curricula),
                sampling=IntegerRandomSampling(),
                crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                eliminate_duplicates=True,
                )
        algorithm = ga

        # prepare the algorithm to solve the specific problem (same arguments as for the minimize function)
        algorithm.setup(curricProblem, termination=('n_gen', self.nGen), seed=1, verbose=False)

        startEpoch, rewards = self.initializeTrainingVariables(os.path.exists(self.logFilePath))
        epoch = startEpoch # 3 x 4

        """
        5x5 aufteilen , 6x6 rausnehmen
        4x4 mit 2x2 obs
        register abhängig von performance machen (ggf x2)
        
        - env schneller machen / laden
        - screen
        - register von performance
        - observation space reduzieren (komplexität des levels wird schwerer; und man kommt durch zufall leichter an) ; nutzlose iterationen sparen
            4, 6, 8, 10 (obs space macht es seeehr schwer)     
        """
        while algorithm.has_next():
            self.txtLogger.info(f"------------------------\nSTART EPOCH {epoch}\n----------------------")
            self.selectedModel = self.args.model + "\\epoch_" + str(epoch)
            pop = algorithm.ask()
            # set biased population for first generation
            if epoch == 1:
                X = self.createFirstGeneration(self.curricula)
                pop = Population.new("X", X)
            self.curricula = self.evolXToCurriculum(pop.get("X"))
            algorithm.evaluator.eval(curricProblem, pop)
            algorithm.tell(infills=pop)
            self.iterationsDone += self.ITERATIONS_PER_ENV  # TODO use exact value
            nextModel = self.args.model + "\\epoch_" + str(epoch + 1)
            currentBestCurriculum = np.argmax(self.currentRewards)
            rewards[str(epoch)] = self.currentRewards
            currentScore = self.currentRewards[currentBestCurriculum]

            utils.copyAgent(
                src=getModelWithCandidatePrefix(utils.getModelName(self.selectedModel, currentBestCurriculum)),
                dest=nextModel)

            self.updateTrainingInfo(epoch, currentBestCurriculum, rewards, currentScore, pop.get("X"))
            self.logRelevantInfo(epoch, currentBestCurriculum)
            epoch += 1

        self.printFinalLogs()
        res = algorithm.result()
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
            self.startTime = datetime.fromisoformat(self.trainingInfoJson["startTime"])
            # delete existing folders, that were created ---> maybe just last one because others should be finished ...
            for k in range(self.args.numberOfCurricula):
                path = self.args.model + "\\epoch_" + str(startEpoch) + "_curric" + str(k)
                if utils.deleteModelIfExists(path):
                    print("deleted", k)
                    snapshotPath = path + "\\_CANDIDATE"
                    utils.deleteModelIfExists(snapshotPath)
                else:
                    print("Nothing to delete", k)
                    break
            # TODO load evol progress
            assert len(self.curricula) == self.trainingInfoJson["curriculaEnvDetails"]["epoch0"]
            self.curricula = self.trainingInfoJson["currentCurriculumList"]
            self.txtLogger.info(f"Continung training from epoch {startEpoch}... ")
        else:
            self.txtLogger.info("Creating model. . .")
            startEpoch = 1
            rewards = self.initializeRewards()  # dict of {"env1": [list of rewards], "env2": [rewards], ...}
            self.iterationsDone = train.main(0, self.selectedModel, ENV_NAMES.DOORKEY_5x5, self.args, self.txtLogger)
            self.initTrainingInfo(rewards, self.iterationsDone)
            utils.copyAgent(src=self.selectedModel, dest=self.args.model + "\\epoch_" + str(
                startEpoch))  # e0 -> e1; subsequent iterations do this the end of each epoch
        return startEpoch, rewards

    def saveTrainingInfoToFile(self):
        """
        Saves the training info into a local file called status.json in the main folder of the model's storage
        """
        with open(self.logFilePath, 'w') as f:
            f.write(json.dumps(self.trainingInfoJson, indent=4, default=str))

    def trainEveryCurriculum(self, curricula):
        """
        This method is called from the curriculumProblem_eval method
        :param curricula: the list of the curricula for the current generaation
        :return: the rewards after the rolling horizon
        """
        rewards = []
        for i in range(len(curricula)):
            rewardI = self.trainEachCurriculum(i, self.iterationsDone)
            rewards.append(rewardI)
        self.currentRewards = rewards
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
            for j in range(curric.shape[0]):
                rounded = round(x[i][j])
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

        The evaluate function receives an integer vector that represents a curriculum. Preferrably, we can use an integer encoding or
        evaluate will return the performance of the RL agent after performing the training with a given curriculum

        the population consists of multiple curricula, which will all be tested

    we initialize an optimization algorithm
    We use the minimize function to run the optimization (minimize will call evaluate and update the population in between)

    we query the result of the minimize function to give us the best curriculum and use the timestep after the first phase of this curriculum as new best RL agent

"""

"""

"""
