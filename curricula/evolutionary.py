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

        self.ITERATIONS_PER_ENV = args.iterationsPerEnv
        self.txtLogger = txtLogger
        self.args = args
        self.startTime = startTime
        # trainingInfoJson & curricula will be initialized in the @startTraining method
        self.trainingInfoJson = {}
        self.curricula = []
        self.logFilePath = os.getcwd() + "\\storage\\" + self.args.model + "\\status.json"
        self.gamma = gamma
        print("init done")

        # self.startCurriculumTraining()

    def trainEachCurriculum(self, i: int, iterationsDone: int, selectedModel: str) -> int:
        """
        Simulates a horizon and returns the rewards obtained after evaluating the state at the end of the horizon
        """
        nameOfCurriculumI = utils.getModelName(selectedModel, i)  # Save TEST_e1 --> TEST_e1_curric0
        reward = 0
        utils.copyAgent(src=selectedModel, dest=nameOfCurriculumI)
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

    def calculateConsecutivelyChosen(self, consecutiveCount, currentBestCurriculum, lastChosenCurriculum) -> int:
        if consecutiveCount + 1 < len(self.curricula[0]) and \
                (currentBestCurriculum == lastChosenCurriculum or lastChosenCurriculum is None):
            return consecutiveCount + 1
        return 0

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
                                 "consecutive": 0,
                                 "startTime": self.startTime,
                                 "numFrames": pretrainingIterations}
        with open(self.logFilePath, 'w') as f:
            f.write(json.dumps(self.trainingInfoJson, indent=4))
        # TODO how expensive is it to always overwrite everything?

    def updateTrainingInfo(self, epoch, iterationsDoneSoFar,
                           currentBestCurriculum,
                           rewards, curriculumConsecutivelyChosen) -> None:
        self.trainingInfoJson["epochsDone"] = epoch + 1
        self.trainingInfoJson["numFrames"] = iterationsDoneSoFar
        curriculumConsecutivelyChosen = curriculumConsecutivelyChosen - 1
        if curriculumConsecutivelyChosen < 0:
            curriculumConsecutivelyChosen = 0

        selectedEnv = self.curricula[currentBestCurriculum][curriculumConsecutivelyChosen]
        self.trainingInfoJson["selectedEnvs"].append(selectedEnv)
        self.trainingInfoJson["bestCurriculaIds"].append(currentBestCurriculum)
        self.trainingInfoJson["rewards"] = rewards
        currentScore = evaluate.evaluateAgent(self.args.model + "\\epoch_" + str(epoch + 1), self.args)
        self.trainingInfoJson["actualPerformance"].append([currentScore, selectedEnv])
        self.trainingInfoJson["curriculaEnvDetails"]["epoch" + str(epoch)] = self.getCurriculaEnvDetails()
        self.trainingInfoJson["consecutive"] = curriculumConsecutivelyChosen
        self.trainingInfoJson["curriculaEnvDetails"]["epoch" + str(epoch + 1)] = self.curricula

        self.txtLogger.info(f"Best results in epoch {epoch} came from curriculum {currentBestCurriculum}")
        self.txtLogger.info(
            f"CurriculaEnvDetails {self.trainingInfoJson['curriculaEnvDetails']}; selectedEnv: {selectedEnv}")
        self.txtLogger.info(f"\nEPOCH: {epoch} SUCCESS\n")

    def startCurriculumTraining(self) -> None:
        """
        Starts The RH Curriculum Training
        """
        iterationsDoneSoFar, startEpoch, rewards, curriculumChosenConsecutivelyTimes, lastChosenCurriculum = \
            self.initializeTrainingVariables(os.path.exists(self.logFilePath))

        for epoch in range(startEpoch, 11):
            selectedModel = self.args.model + "\\epoch_" + str(epoch)
            for i in range(len(self.curricula)):
                reward = self.trainEachCurriculum(i, iterationsDoneSoFar, selectedModel)
                rewards[str(i)].append(reward)
            iterationsDoneSoFar += self.ITERATIONS_PER_ENV  # TODO use exact value; maybe return something during training
            currentBestCurriculum = int(
                np.argmax([lst[-1] for lst in rewards.values()]))  # only access the latest reward

            utils.copyAgent(src=getModelWithCandidatePrefix(utils.getModelName(selectedModel, currentBestCurriculum)),
                            dest=self.args.model + "\\epoch_" + str(epoch + 1))  # the model for the next epoch

            curriculumChosenConsecutivelyTimes = \
                self.calculateConsecutivelyChosen(curriculumChosenConsecutivelyTimes, currentBestCurriculum,
                                                  lastChosenCurriculum)
            lastChosenCurriculum = currentBestCurriculum

            self.updateTrainingInfo(epoch, iterationsDoneSoFar, currentBestCurriculum, rewards,
                                    curriculumChosenConsecutivelyTimes)
            self.updateCurriculaAfterHorizon(self.curricula[currentBestCurriculum], self.args.numberOfCurricula)
            self.trainingInfoJson["currentCurriculumList"] = self.curricula  # TODO refactor this
            self.saveTrainingInfoToFile()

        self.printFinalLogs()

    @staticmethod
    def initializeCurricula(numberOfCurricula: int, envsPerCurriculum: int) -> list:
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

    def updateCurriculaAfterHorizon(self, bestCurriculum: list, numberOfCurricula: int) -> None:
        """
        Updates the List of curricula by using the last N-1 Envs, and randomly selecting a last new one
        :param numberOfCurricula:
        :param bestCurriculum: full env list of the curriculum that performed best during last epoch
                (i.e. needs to be cut by 1 element!)
        """
        self.curricula = []
        for i in range(numberOfCurricula):
            self.curricula.append(bestCurriculum[1:])
            val = np.random.randint(0, len(ENV_NAMES.ALL_ENVS))
            self.curricula[i].append(ENV_NAMES.ALL_ENVS[val])
        # TODO remove duplicates
        # TODO dealing with the case where duplicates are forced due to parameters
        # TODO maybe only do this for a percentage of curricula, and randomly set the others OR instead of using [1:], use [1:__]
        assert len(self.curricula) == numberOfCurricula

    def initializeTrainingVariables(self, modelExists) -> tuple:
        """
        Initializes and returns all the necessary training variables
        :param modelExists: whether the path to the model already exists or not
        """
        if modelExists:
            with open(self.logFilePath, 'r') as f:
                self.trainingInfoJson = json.loads(f.read())

            iterationsDoneSoFar = self.trainingInfoJson["numFrames"]
            startEpoch = self.trainingInfoJson["epochsDone"]
            rewards = self.trainingInfoJson["rewards"]
            curriculumChosenConsecutivelyTimes = self.trainingInfoJson["consecutive"]
            lastChosenCurriculum = self.trainingInfoJson["bestCurriculaIds"][-1]
            self.curricula = self.trainingInfoJson["currentCurriculumList"]
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
            self.curricula = self.initializeCurricula(self.args.numberOfCurricula, self.args.envsPerCurriculum)
            rewards = self.initializeRewards()  # dict of {"env1": [list of rewards], "env2": [rewards], ...}
            selectedModel = self.args.model + "\\epoch_" + str(0)
            iterationsDoneSoFar = train.main(0, selectedModel, ENV_NAMES.DOORKEY_5x5, self.args, self.txtLogger)
            self.initTrainingInfo(rewards, iterationsDoneSoFar)
            utils.copyAgent(src=selectedModel, dest=self.args.model + "\\epoch_" + str(
                startEpoch))  # e0 -> e1; subsequent iterations do at the end of each epoch iteration
            curriculumChosenConsecutivelyTimes = 0
            lastChosenCurriculum = None
        return iterationsDoneSoFar, startEpoch, rewards, curriculumChosenConsecutivelyTimes, lastChosenCurriculum

    def saveTrainingInfoToFile(self):
        with open(self.logFilePath, 'w') as f:
            f.write(json.dumps(self.trainingInfoJson, indent=4))

    def trainEveryCurriculum(self, curricula, selectedModel):
        iterationsDone = 0
        rewards = []  # TODO CHECK IF THIS IS THE CORRECT D-TYPE FOR PYMOO
        for i in range(len(curricula)):
            rewardI = self.trainEachCurriculum(i, iterationsDone, selectedModel)
            rewards.append(rewardI)
        return rewards


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


def tryEvolStuff(txtLogger, startTime, args):
    objectives = 1
    curric1 = [ENV_NAMES.DOORKEY_5x5, ENV_NAMES.DOORKEY_5x5, ENV_NAMES.DOORKEY_6x6, ENV_NAMES.DOORKEY_6x6]
    curric2 = [ENV_NAMES.DOORKEY_8x8, ENV_NAMES.DOORKEY_16x16, ENV_NAMES.DOORKEY_16x16, ENV_NAMES.DOORKEY_16x16]
    xupper = len(ENV_NAMES.ALL_ENVS)
    curriculaList = [curric2, curric1]
    inequalityConstr = 0
    rhEvol = EvolutionaryCurriculum(txtLogger, startTime, args)
    curricProblem = CurriculumProblem(curriculaList, objectives, inequalityConstr, xupper, rhEvol)

    algorithm = NSGA2(pop_size=len(curriculaList),
                      # sampling=BinaryRandomSampling(),
                      # crossover=TwoPointCrossover(),
                      # mutation=BitflipMutation(),
                      eliminate_duplicates=True,
                      )

    # prepare the algorithm to solve the specific problem (same arguments as for the minimize function)
    algorithm.setup(curricProblem, termination=('n_gen', 10), seed=1, verbose=False) # TODO args. ...
    X = createFirstGeneration(curriculaList) # todo only do on first load
    initialPop = Population.new("X", X)
    iterationsDoneSoFar, startEpoch, rewards, curriculumChosenConsecutivelyTimes, lastChosenCurriculum = \
        rhEvol.initializeTrainingVariables(os.path.exists(rhEvol.logFilePath))

    while algorithm.has_next():
        if initialPop is not None:
            # ask the algorithm for the next solution to be evaluated
            pop = algorithm.ask()
        else:
            pop = initialPop
            initialPop = None
        # evaluate the individuals using the algorithm's evaluator (necessary to count evaluations for termination)
        algorithm.evaluator.eval(curricProblem, pop)
        # returned the evaluated individuals which have been evaluated or even modified

        algorithm.tell(infills=pop)
        # do same more things, printing, logging, storing or even modifying the algorithm object
        # print("n_gen, n_eval:", algorithm.n_gen, algorithm.evaluator.n_eval)
        print("----", "pop X:", np.round(pop.get("X")), pop.get("F"), "-----", sep="\n")
        currentBestCurriculum = int(
            np.argmax([lst[-1] for lst in rewards.values()]))  # only access the latest reward

        utils.copyAgent(src=getModelWithCandidatePrefix(utils.getModelName(selectedModel, currentBestCurriculum)),
                        dest=self.args.model + "\\epoch_" + str(self.N + 1))  # the model for the next epoch

        lastChosenCurriculum = currentBestCurriculum

        rhEvol.updateTrainingInfo(epoch, iterationsDoneSoFar, currentBestCurriculum, rewards,
                                  curriculumChosenConsecutivelyTimes)
        rhEvol.trainingInfoJson["currentCurriculumList"] = self.curricula  # use evolution.X or sth
        rhEvol.saveTrainingInfoToFile()

    rhEvol.printFinalLogs()
        # TODO update the model with best individual
        iterationsDoneSoFar += self.ITERATIONS_PER_ENV  # TODO use exact value; maybe return something during training
        #         curriculumChosenConsecutivelyTimes = \
        #             self.calculateConsecutivelyChosen(curriculumChosenConsecutivelyTimes, currentBestCurriculum,
        #                                               lastChosenCurriculum) # TODO can probably remove it now
    res = algorithm.result()
    print("hash", res.F.sum())
    print(np.round(res.X))

    return 0


def createFirstGeneration(curriculaList):
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
        # maybe try to avoid homogenous curricula with ieq constraints (?)
        xl = np.full(n_var, -0.49)
        xu = np.full(n_var, xu - 0.51, dtype=float)
        super().__init__(n_var=n_var,  # the amount of curricula to use
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
        curricula = self.evolXToCurriculum(x)
        selectedModel = self.args.model + "\\epoch_" + str(self.N)
        rewards = self.evolCurric.trainEveryCurriculum(curricula, selectedModel)
        reward = self.trainEachCurriculum(i, iterationsDoneSoFar, selectedModel)
        rewards[str(i)].append(reward)
        out["F"] = -1 * rewards
        self.N += 1

    @staticmethod
    def evolXToCurriculum(x):
        result = []
        for i in range(x.shape[0]):
            result.append([])
            curric = x[i]
            for j in range(curric.shape[0]):
                rounded = round(x[i][j])
                result[i].append(ENV_NAMES.ALL_ENVS[rounded])
        return result
