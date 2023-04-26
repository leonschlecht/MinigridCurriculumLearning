import json
import os
from datetime import datetime
import numpy as np
import utils
from curricula.CurriculumProblem import CurriculumProblem
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

        objectives = 1
        curric1 = [ENV_NAMES.DOORKEY_5x5, ENV_NAMES.DOORKEY_5x5, ENV_NAMES.DOORKEY_16x16, ENV_NAMES.DOORKEY_6x6]
        curric2 = [ENV_NAMES.DOORKEY_8x8, ENV_NAMES.DOORKEY_16x16, ENV_NAMES.DOORKEY_16x16, ENV_NAMES.DOORKEY_16x16]
        xupper = len(ENV_NAMES.ALL_ENVS)
        self.curricula = [curric1, curric2]
        inequalityConstr = 0

        self.ITERATIONS_PER_ENV = args.iterationsPerEnv
        self.iterationsDone = 0
        self.txtLogger = txtLogger
        self.args = args
        self.startTime = startTime
        self.selectedModel = args.model + "\\epoch_0"
        self.nGen = args.curriculumEpochs

        # trainingInfoJson & curricula will be initialized in the @startTrainingLoop method
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
        return fullEnvList

    def printFinalLogs(self) -> None:
        """
        Prints the last logs, after the training is done
        """
        self.txtLogger.info("----TRAINING END-----")
        self.txtLogger.info(f"Best Curricula {self.trainingInfoJson['bestCurriculaIds']}")
        self.txtLogger.info(f"Trained in Envs {self.trainingInfoJson['selectedEnvs']}")
        self.txtLogger.info(f"Rewards: {self.trainingInfoJson['rewards']}")
        print(self.startTime)
        # startTime = datetime.strptime(self.startTime, '%Y-%m-%d %H:%M:%S') # TODO fix

        now = datetime.now()
        timeDiff = self.startTime - now
        # print(timeDiff)
        # self.txtLogger.info(f"Time ended at {now} , total training time: {timeDiff}")
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

    def logRelevantInfo(self, epoch, currentBestCurriculum):
        """
        Logs relevant training info after a training epoch is done and the trainingInfo was already updated
        :param epoch:
        :param currentBestCurriculum:
        :return:
        """
        selectedEnv = self.trainingInfoJson["selectedEnvs"][-1]
        envDetailsOfCurrentEpoch = self.getCurriculaEnvDetails() # TODO check if this is from the right iteration

        self.txtLogger.info(f"Best results in epoch {epoch} came from curriculum {currentBestCurriculum}")
        self.txtLogger.info(
            f"CurriculaEnvDetails {envDetailsOfCurrentEpoch}; selectedEnv: {selectedEnv}")
        self.txtLogger.info(f"\nEPOCH: {epoch} SUCCESS\n")

    def startTrainingLoop(self, objectives: int, inequalityConstr, xupper):
        curricProblem = CurriculumProblem(self.curricula, objectives, inequalityConstr, xupper, self)

        algorithm = NSGA2(pop_size=len(self.curricula),
                          sampling=IntegerRandomSampling(),
                          crossover=SBX(),
                          mutation=PM(),
                          eliminate_duplicates=True,
                          )
        # NSGA Default:
        # sampling: FloatRandomSampling = FloatRandomSampling(),
        # selection: TournamentSelection = TournamentSelection(func_comp=binary_tournament),
        # crossover: SBX = SBX(eta=15, prob=0.9),
        # mutation: PM = PM(eta=20),

        # prepare the algorithm to solve the specific problem (same arguments as for the minimize function)
        algorithm.setup(curricProblem, termination=('n_gen', self.nGen), seed=1, verbose=False)  # TODO args. for n_gen
        X = createFirstGeneration(self.curricula)  # todo only do on first load
        initialPop = Population.new("X", X)
        print("initialPop =", initialPop)
        startEpoch, rewards = self.initializeTrainingVariables(os.path.exists(self.logFilePath))
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
            self.currentRewards = [1]
            currentBestCurriculum = np.argmax(self.currentRewards)
            rewards[str(epoch)] = [self.currentRewards]

            # utils.copyAgent( TODO uncomment
            #    src=getModelWithCandidatePrefix(utils.getModelName(self.selectedModel, currentBestCurriculum)),
            #   dest=nextModel)

            currentScore = -10
            # currentScore = evaluate.evaluateAgent(self.args.model + "\\epoch_" + str(epoch + 1), self.args)

            # self.updateTrainingInfo(epoch, currentBestCurriculum, rewards, currentScore)
            self.logRelevantInfo(epoch, currentBestCurriculum)
            # TODO split update & log into 2 methods
            epoch += 1

        # self.printFinalLogs()
        res = algorithm.result()
        print("final fitness:", res.F.sum())
        print("Final X = ", res.X)

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
            # self.startTime = self.trainingInfoJson["startTime"] # TODO convert
            # delete existing folders, that were created ---> maybe just last one because others should be finished ...
            for k in range(self.args.numberOfCurricula):
                # TODO test this
                path = self.logFilePath + "\\epoch" + str(k)
                if os.path.exists(path):
                    utils.deleteModel(path)
                    utils.deleteModel(path + "\\_CANDIDATE")
                else:
                    break
            # assert len(self.curricula) == self.trainingInfoJson["curriculaEnvDetails"]["epoch0"]
            self.txtLogger.info(f"Continung training from epoch {startEpoch}... ")
        else:
            self.txtLogger.info("Creating model. . .")
            startEpoch = 1
            rewards = self.initializeRewards()  # dict of {"env1": [list of rewards], "env2": [rewards], ...}
            self.iterationsDone = train.main(0, self.selectedModel, ENV_NAMES.DOORKEY_5x5, self.args, self.txtLogger)
            self.initTrainingInfo(rewards, self.iterationsDone)
            utils.copyAgent(src=self.selectedModel, dest=self.args.model + "\\epoch_" + str(
                startEpoch))  # e0 -> e1; subsequent iterations do this the end of each epoch
        # TODO initialize self.curricula here
        return startEpoch, rewards

    def saveTrainingInfoToFile(self):
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
    :return the transformed list containing integers representing the environment Nr
    """
    indices = []
    for i in range(len(curriculaList)):
        indices.append([])
        for env in curriculaList[i]:
            indices[i].append(ENV_NAMES.ALL_ENVS.index(env))
    return indices


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
