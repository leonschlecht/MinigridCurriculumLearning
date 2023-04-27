import json
import os
import time

import numpy as np

import utils
from scripts import train, evaluate
from utils import ENV_NAMES, getModelWithCandidatePrefix


class BiasedRandomRollingHorizon:
    """
    This class represents a "biased" random rolling horizon. I.e not fully random (like the class
    FullRandomHorizon.
    The difference is that the best curriculum is the base for the next iteration for x% of the curricula in the list
    This aims to achieve that given the last best curriculum, based on this we want to continue most of our training.
    """

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

        self.startCurriculumTraining()

    def startCurriculumTraining(self) -> None:
        """
        Starts The RH Curriculum Training
        """
        iterationsDoneSoFar, startEpoch, rewards, lastChosenCurriculum = \
            self.initializeTrainingVariables(os.path.exists(self.logFilePath))

        for epoch in range(startEpoch, 11):
            selectedModel = self.args.model + "\\epoch_" + str(epoch)
            for i in range(len(self.curricula)):
                reward = self.trainEachCurriculum(i, iterationsDoneSoFar, selectedModel)
                rewards[str(i)].append(reward)
            iterationsDoneSoFar += self.ITERATIONS_PER_ENV  # TODO use exact value; maybe return something during training
            currentBestCurriculum = int(
                np.argmax([lst[-1] for lst in rewards.values()]))  # only access the latest reward

            utils.copyAgent(
                src=getModelWithCandidatePrefix(utils.getModelName(selectedModel, currentBestCurriculum)),
                dest=self.args.model + "\\epoch_" + str(epoch + 1))  # the model for the next epoch

            self.updateTrainingInfo(epoch, iterationsDoneSoFar, currentBestCurriculum, rewards)
            self.curricula = self.initializeCurricula(self.args.numberOfCurricula, self.args.envsPerCurriculum)
            self.trainingInfoJson["currentCurriculumList"] = self.curricula  # TODO refactor this
            self.saveTrainingInfoToFile()

        self.printFinalLogs()

    def trainEachCurriculum(self, i, iterationsDone, selectedModel) -> int:
        """
        Simulates a horizon and returns the rewards obtained after evaluating the state at the end of the horizon
        """
        nameOfCurriculumI = utils.getModelName(selectedModel, i)  # Save TEST_e1 --> TEST_e1_curric0
        rewards = 0
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
        self.txtLogger.info(f"Rewards for curriculum {nameOfCurriculumI} = {rewards}")
        return rewards

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
        self.txtLogger.info(f"Time ended at {time.time()} , total training time: {time.time() - self.startTime}")
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
        with open(self.logFilePath, 'w') as f:
            f.write(json.dumps(self.trainingInfoJson, indent=4))
        # TODO how expensive is it to always overwrite everything?

    def updateTrainingInfo(self, epoch, iterationsDoneSoFar,
                           currentBestCurriculum, rewards) -> None:
        self.trainingInfoJson["epochsDone"] = epoch + 1
        self.trainingInfoJson["numFrames"] = iterationsDoneSoFar

        selectedEnv = self.curricula[currentBestCurriculum][0]
        self.trainingInfoJson["selectedEnvs"].append(selectedEnv)
        self.trainingInfoJson["bestCurriculaIds"].append(currentBestCurriculum)
        self.trainingInfoJson["rewards"] = rewards
        currentScore = evaluate.evaluateAgent(self.args.model + "\\epoch_" + str(epoch + 1), self.args)
        self.trainingInfoJson["actualPerformance"].append([currentScore, selectedEnv])
        self.trainingInfoJson["curriculaEnvDetails"]["epoch" + str(epoch)] = self.getCurriculaEnvDetails()
        self.trainingInfoJson["curriculaEnvDetails"]["epoch" + str(epoch + 1)] = self.curricula

        self.txtLogger.info(f"Best results in epoch {epoch} came from curriculum {currentBestCurriculum}")
        self.txtLogger.info(
            f"CurriculaEnvDetails {self.trainingInfoJson['curriculaEnvDetails']}; selectedEnv: {selectedEnv}")
        self.txtLogger.info(f"\nEPOCH: {epoch} SUCCESS\n")

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
            # TODO find better way instead of calling train.main to create folder
            self.txtLogger.info("Creating model. . .")
            startEpoch = 1
            self.curricula = self.initializeCurricula(self.args.numberOfCurricula, self.args.envsPerCurriculum)
            rewards = self.initializeRewards()  # dict of {"env1": [list of rewards], "env2": [rewards], ...}
            selectedModel = self.args.model + "\\epoch_" + str(0)
            iterationsDoneSoFar = train.main(0, selectedModel, ENV_NAMES.DOORKEY_5x5, self.args, self.txtLogger)
            self.initTrainingInfo(rewards, iterationsDoneSoFar)
            utils.copyAgent(src=selectedModel, dest=self.args.model + "\\epoch_" + str(
                startEpoch))  # e0 -> e1; subsequent iterations do at the end of each epoch iteration
            lastChosenCurriculum = None
        return iterationsDoneSoFar, startEpoch, rewards, lastChosenCurriculum

    def saveTrainingInfoToFile(self):
        with open(self.logFilePath, 'w') as f:
            f.write(json.dumps(self.trainingInfoJson, indent=4))


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
