import json
import os
import time

import numpy as np

import utils
from scripts import train, evaluate
from utils import ENV_NAMES, getModelWithCandidatePrefix
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.core.problem import Problem


class EvolutionaryCurriculum:

    def __init__(self, txtLogger, startTime, args, gamma=.9):
        assert args.envsPerCurriculum > 0
        assert args.numberOfCurricula > 0
        assert args.iterationsPerEnv > 0

        objectives = 1
        curric1 = [ENV_NAMES.DOORKEY_5x5, ENV_NAMES.DOORKEY_5x5]
        curric2 = [ENV_NAMES.DOORKEY_8x8, ENV_NAMES.DOORKEY_16x16, ENV_NAMES.DOORKEY_16x16, ENV_NAMES.DOORKEY_16x16]
        xupper = len(ENV_NAMES.ALL_ENVS)
        self.curricula = [curric1]
        inequalityConstr = 0

        self.ITERATIONS_PER_ENV = args.iterationsPerEnv
        self.iterationsDone = 0
        self.txtLogger = txtLogger
        self.args = args
        self.startTime = startTime
        self.selectedModel = self.args.model + "\\epoch_0"

        # trainingInfoJson & curricula will be initialized in the @startTraining method
        self.trainingInfoJson = {}
        self.logFilePath = os.getcwd() + "\\storage\\" + self.args.model + "\\status.json"
        self.gamma = gamma
        self.currentRewards = []

        self.startTrainingLoop(objectives, inequalityConstr, xupper)
        # self.startCurriculumTraining()

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
            self.txtLogger.info(f"Iterations Done {iterationsDone}")
            if j == 0:
                utils.copyAgent(src=nameOfCurriculumI, dest=utils.getModelWithCandidatePrefix(
                    nameOfCurriculumI))  # save TEST_e1_curric0 -> + _CANDIDATE
                # TODO save reward separately here?
            self.txtLogger.info(f"Trained iteration j={j} of curriculum {nameOfCurriculumI} ")
            # rewards += ((self.gamma ** j) * evaluate.evaluateAgent(nameOfCurriculumI, self.args))  # TODO or (j+1) ?
        self.txtLogger.info(f"Reward for curriculum {nameOfCurriculumI} = {reward}")
        return i  # reward TODO

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
        print("full env list", fullEnvList)
        return fullEnvList

    def printFinalLogs(self) -> None:
        """
        Prints the last logs, after the training is done
        """
        self.txtLogger.info("----TRAINING END-----")
        self.txtLogger.info(f"Best Curricula {self.trainingInfoJson['bestCurriculaIds']}")
        self.txtLogger.info(f"Trained in Envs {self.trainingInfoJson['selectedEnvs']}")
        self.txtLogger.info(f"Rewards: {self.trainingInfoJson['rewards']}")
        self.txtLogger.info(f"Time ended at {time.time()} , total training time: {time.time() - self.startTime}")
        self.txtLogger.info("-------------------\n\n")

    def initTrainingInfo(self, rewards, pretrainingIterations):
        self.trainingInfoJson = {"selectedEnvs": [],
                                 "bestCurriculaIds": [],
                                 "curriculaEnvDetails": {},
                                 "rewards": rewards,
                                 "actualPerformance": [],
                                 "epochsDone": 1,
                                 "startTime": self.startTime,  # TODO convert datetime object to actual Date
                                 "numFrames": pretrainingIterations}
        print(self.trainingInfoJson)
        self.saveJsonFile(self.logFilePath, self.trainingInfoJson)
        # TODO how expensive is it to always overwrite everything?

    @staticmethod
    def saveJsonFile(path, jsonBody):
        with open(path, 'w') as f:
            f.write(json.dumps(jsonBody, indent=4, default=str))

    def updateTrainingInfo(self, epoch, currentBestCurriculum, rewards) -> None:
        self.trainingInfoJson["epochsDone"] = epoch + 1
        self.trainingInfoJson["numFrames"] = self.iterationsDone

        selectedEnv = self.curricula[currentBestCurriculum][0]
        self.trainingInfoJson["selectedEnvs"].append(selectedEnv)
        self.trainingInfoJson["bestCurriculaIds"].append(currentBestCurriculum)
        self.trainingInfoJson["rewards"] = rewards
        currentScore = evaluate.evaluateAgent(self.args.model + "\\epoch_" + str(epoch + 1), self.args)
        self.trainingInfoJson["actualPerformance"].append([currentScore, selectedEnv])
        envDetailsOfCurrentEpoch = self.getCurriculaEnvDetails()
        self.trainingInfoJson["curriculaEnvDetails"]["epoch" + str(epoch)] = envDetailsOfCurrentEpoch
        self.trainingInfoJson["curriculaEnvDetails"]["epoch" + str(epoch + 1)] = self.curricula  # save as backup

        self.txtLogger.info(f"Best results in epoch {epoch} came from curriculum {currentBestCurriculum}")
        self.txtLogger.info(
            f"CurriculaEnvDetails {envDetailsOfCurrentEpoch}; selectedEnv: {selectedEnv}")
        self.txtLogger.info(f"\nEPOCH: {epoch} SUCCESS\n")

    def startTrainingLoop(self, objectives: int, inequalityConstr, xupper):
        curricProblem = CurriculumProblem(self.curricula, objectives, inequalityConstr, xupper, self)

        algorithm = NSGA2(pop_size=len(self.curricula),
                          # sampling=BinaryRandomSampling(),
                          # crossover=TwoPointCrossover(),
                          # mutation=BitflipMutation(),
                          eliminate_duplicates=True,
                          )  # TODO use Integer crossover? etc

        # prepare the algorithm to solve the specific problem (same arguments as for the minimize function)
        algorithm.setup(curricProblem, termination=('n_gen', 10), seed=1, verbose=False)  # TODO args. for n_gen
        X = createFirstGeneration(self.curricula)  # todo only do on first load
        initialPop = Population.new("X", X)
        startEpoch, rewards, lastChosenCurriculum = self.initializeTrainingVariables(os.path.exists(self.logFilePath))
        epoch = startEpoch
        while algorithm.has_next():
            self.txtLogger.info(f"------------------------\nSTART EPOCH {epoch}\n----------------------")
            self.selectedModel = self.args.model + "\\epoch_" + str(epoch)
            pop = algorithm.ask()
                # pop = initialPop
                # initialPop = None
            self.curricula = self.evolXToCurriculum(pop.get("X"))
            print(self.curricula)
            algorithm.evaluator.eval(curricProblem, pop)
            algorithm.tell(infills=pop)
            self.iterationsDone += self.ITERATIONS_PER_ENV  # TODO use exact value
            nextModel = self.args.model + "\\epoch_" + str(epoch + 1)
            currentBestCurriculum = np.argmax(self.currentRewards)
            rewards[str(epoch)] = [self.currentRewards]

            utils.copyAgent(
                src=getModelWithCandidatePrefix(utils.getModelName(self.selectedModel, currentBestCurriculum)),
                dest=nextModel)

            self.updateTrainingInfo(epoch, currentBestCurriculum, rewards)
            self.trainingInfoJson["currentCurriculumList"] = self.evolXToCurriculum(pop.get("X"))
            self.trainingInfoJson["curriculumListAsX"] = pop.get("X")
            self.saveTrainingInfoToFile()
            epoch += 1

        self.printFinalLogs()
        res = algorithm.result()
        print("hash", res.F.sum())
        print(np.round(res.X))

    @staticmethod
    def randomlyInitializeCurricula(numberOfCurricula: int, envsPerCurriculum: int) -> list:
        """
        Initializes the list of Curricula randomly
        :param numberOfCurricula: how many curricula will be generated
        :param envsPerCurriculum: how many environment each curriculum has
        """
        curricula = []
        i = 0
        while i < numberOfCurricula:
            newCurriculum = []
            for j in range(envsPerCurriculum):
                val = np.random.randint(0, len(ENV_NAMES.ALL_ENVS))
                newCurriculum.append(ENV_NAMES.ALL_ENVS[val])
            if newCurriculum not in curricula:  # TODO find better duplicate checking method
                curricula.append(newCurriculum)
                i += 1
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
            lastChosenCurriculum = self.trainingInfoJson["bestCurriculaIds"][-1]
            self.startTime = self.trainingInfoJson["startTime"]
            # delete existing folders, that were created ---> maybe just last one because others should be finished ...
            for k in range(self.args.numberOfCurricula):
                # TODO test this
                path = self.logFilePath + "\\epoch" + str(k)
                if os.path.exists(path):
                    utils.deleteModel(path)
                    utils.deleteModel(path + "\\_CANDIDATE")
                else:
                    break
            assert len(self.curricula) == self.trainingInfoJson["curriculaEnvDetails"]["epoch0"]
            self.txtLogger.info(f"Continung training from epoch {startEpoch}... ")
        else:
            self.txtLogger.info("Creating model. . .")
            startEpoch = 1
            rewards = self.initializeRewards()  # dict of {"env1": [list of rewards], "env2": [rewards], ...}
            self.iterationsDone = train.main(0, self.selectedModel, ENV_NAMES.DOORKEY_5x5, self.args, self.txtLogger)
            self.initTrainingInfo(rewards, self.iterationsDone)
            utils.copyAgent(src=self.selectedModel, dest=self.args.model + "\\epoch_" + str(
                startEpoch))  # e0 -> e1; subsequent iterations do this the end of each epoch
            lastChosenCurriculum = None
        # TODO initialize self.curricula here
        return startEpoch, rewards, lastChosenCurriculum

    def saveTrainingInfoToFile(self):
        with open(self.logFilePath, 'w') as f:
            f.write(json.dumps(self.trainingInfoJson, indent=4, default=str))

    def trainEveryCurriculum(self, curricula):
        rewards = []  # TODO CHECK IF THIS IS THE CORRECT D-TYPE FOR PYMOO
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


def createFirstGeneration(curriculaList):
    """
    Helper method that creates the biased first population of the RHEA
    Transform from environment language string -> numbers
    :param curriculaList:
    :return:
    """
    indices = []
    for i in range(len(curriculaList)):
        indices.append([])
        for env in curriculaList[i]:
            indices[i].append(ENV_NAMES.ALL_ENVS.index(env))
    return indices


class CurriculumProblem(Problem):
    def __init__(self, curricula: list, n_obj, n_ieq_constr, xu, evolCurric: EvolutionaryCurriculum):
        assert len(curricula) > 0
        assert evolCurric is not None
        self.evolCurric = evolCurric
        n_var = len(curricula[0])
        # TODO maybe try to avoid homogenous curricula with ieq constraints (?)
        xl = np.full(n_var, -0.49)
        xu = np.full(n_var, xu - 0.51, dtype=float)
        super().__init__(n_var=n_var,
                         n_obj=n_obj,  # maximizing the overall reward = 1 objective
                         n_ieq_constr=n_ieq_constr,  # 0 ?
                         xl=xl,
                         xu=xu)
        self.curricula = curricula
        self.N = 0

        # F: what we want to maximize: ---> pymoo minimizes, so it should be -reward
        # G:# Inequality constraint;
        # H is EQ constraint: maybe we can experiment with the length of each curriculum;
        #   and maybe with iterations_per_env (so that each horizon has same length still)

    def _evaluate(self, x, out, *args, **kwargs):
        curricula = self.evolCurric.evolXToCurriculum(x)
        rewards = self.evolCurric.trainEveryCurriculum(curricula)
        out["F"] = -1 * rewards
        print("EVALUATE PYMOO DONE")
