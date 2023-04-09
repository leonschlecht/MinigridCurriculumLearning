import json
import os
import time

import numpy as np

import utils
from scripts import train, evaluate
from utils import ENV_NAMES, getModelWithCandidatePrefix


class EvolutionaryCurriculum:

    def __init__(self, ITERATIONS_PER_ENV: int, txtLogger, startTime, curricula: list, args, gamma=.9):
        self.ITERATIONS_PER_ENV = ITERATIONS_PER_ENV
        self.txtLogger = txtLogger
        self.args = args
        self.startTime = startTime
        self.curricula = curricula
        self.trainingInfoJson = {}
        assert len(curricula) > 0
        self.logFilePath = os.getcwd() + "\\storage\\" + self.args.model + "\\status.json"
        self.gamma = gamma

        self.startCurriculumTraining()

    def trainEachCurriculum(self, i, iterationsDone, selectedModel, startingIndex) -> int:
        """
        Simulates a horizon and returns the rewards obtained after evaluating the state at the end of the horizon
        """
        nameOfCurriculumI = utils.getModelName(selectedModel, i)  # Save TEST_e1 --> TEST_e1_curric0
        rewards = 0
        utils.copyAgent(src=selectedModel, dest=nameOfCurriculumI)
        for j in range(startingIndex, len(self.curricula[i])):
            iterationsDone = train.main(iterationsDone + self.ITERATIONS_PER_ENV, nameOfCurriculumI,
                                        self.curricula[i][j], self.args, self.txtLogger)
            self.txtLogger.info(f"Iterations Done {iterationsDone}")
            if j == startingIndex:
                utils.copyAgent(src=nameOfCurriculumI,
                                dest=utils.getModelWithCandidatePrefix(
                                    nameOfCurriculumI))  # save TEST_e1_curric0 -> + _CANDIDATE
                # TODO save reward separately here?
            self.txtLogger.info(f"Trained iteration j {j} (offset {startingIndex}) of curriculum {i}")
            rewards += self.gamma ** i * evaluate.evaluateAgent(nameOfCurriculumI, self.args)  # TODO or (i+1) ?
        return rewards

    def initializeRewards(self):
        """
        Loads the rewards given the state from previous training, or empty initializes them if first time run
        This assumes that the curricula are numbered from 1 to N (as opposed to using their names or something similar)
        """
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

    def calculateConsecutivelyChosen(self, consecutiveCount, currentBestCurriculum, lastChosenCurriculum,
                                     curriculaList) -> int:
        if consecutiveCount + 1 >= len(curriculaList[0]) or currentBestCurriculum != lastChosenCurriculum:
            return 0
        return consecutiveCount + 1

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
                                 "numFrames": pretrainingIterations}
        with open(self.logFilePath, 'w') as f:
            f.write(json.dumps(self.trainingInfoJson, indent=4))

    def updateTrainingInfo(self, epoch, iterationsDoneSoFar, selectedEnv,
                           currentBestCurriculum,
                           rewards, curriculaEnvDetails, curriculumConsecutivelyChosen) -> None:
        self.trainingInfoJson["epochsDone"] = epoch + 1
        self.trainingInfoJson["numFrames"] = iterationsDoneSoFar
        self.trainingInfoJson["selectedEnvs"].append(
            selectedEnv)  # TODO Test if this works properly / maybe remove bcs redundant
        self.trainingInfoJson["bestCurriculaIds"].append(currentBestCurriculum)
        self.trainingInfoJson["rewards"] = rewards
        currentScore = evaluate.evaluateAgent(self.args.model + "\\epoch_" + str(epoch + 1), self.args)
        self.trainingInfoJson["actualPerformance"].append([currentScore, selectedEnv])
        self.trainingInfoJson["curriculaEnvDetails"]["epoch" + str(epoch)] = curriculaEnvDetails
        self.trainingInfoJson["consecutive"] = curriculumConsecutivelyChosen

        with open(self.logFilePath, 'w') as f:
            f.write(json.dumps(self.trainingInfoJson, indent=4))

        self.txtLogger.info(f"Best results in epoch {epoch} came from curriculum {currentBestCurriculum}")
        self.txtLogger.info(f"CurriculaEnvDetails {self.trainingInfoJson['curriculaEnvDetails']}")
        self.txtLogger.info(f"\nEPOCH: {epoch} SUCCESS\n")

    def calculateStartingIndex(self, curriculumChosenConsecutivelyTimes, isZero) -> int:
        if isZero:
            return 0
        return curriculumChosenConsecutivelyTimes

    def startCurriculumTraining(self) -> None:
        """
        Starts The RH Curriculum Training
        """
        if os.path.exists(self.logFilePath):
            with open(self.logFilePath, 'r') as f:
                self.trainingInfoJson = json.loads(f.read())

            iterationsDoneSoFar = self.trainingInfoJson["numFrames"]
            startEpoch = self.trainingInfoJson["epochsDone"]
            rewards = self.trainingInfoJson["rewards"]
            curriculumChosenConsecutivelyTimes = self.trainingInfoJson["consecutive"]
            lastChosenCurriculum = self.trainingInfoJson["bestCurriculaIds"][-1]

            self.txtLogger.info(f"Continung training from epoch {startEpoch}... ")
        else:
            startEpoch = 1
            rewards = self.initializeRewards()  # dict of {"env1": [list of rewards], "env2": [rewards], ...}
            selectedModel = self.args.model + "\\epoch_" + str(0)
            self.txtLogger.info("Pretraining. . .")
            iterationsDoneSoFar = train.main(self.ITERATIONS_PER_ENV, selectedModel, ENV_NAMES.DOORKEY_5x5, self.args,
                                             self.txtLogger)
            self.initTrainingInfo(rewards, iterationsDoneSoFar)
            utils.copyAgent(src=selectedModel, dest=self.args.model + "\\epoch_" + str(
                startEpoch))  # e0 -> e1; subsequent iterations do at the end of each epoch iteration
            curriculumChosenConsecutivelyTimes = 0
            lastChosenCurriculum = None

        for epoch in range(startEpoch, 11):
            selectedModel = self.args.model + "\\epoch_" + str(epoch)
            for i in range(len(self.curricula)):
                startingIndex = self.calculateStartingIndex(curriculumChosenConsecutivelyTimes,
                                                            i == lastChosenCurriculum)
                reward = self.trainEachCurriculum(i, iterationsDoneSoFar, selectedModel, startingIndex)
                rewards[str(i)].append(reward)
            iterationsDoneSoFar += self.ITERATIONS_PER_ENV
            currentBestCurriculum = int(
                np.argmax([lst[-1] for lst in rewards.values()]))  # only access the latest reward

            utils.copyAgent(src=getModelWithCandidatePrefix(utils.getModelName(selectedModel, currentBestCurriculum)),
                            dest=self.args.model + "\\epoch_" + str(epoch + 1))  # the model for the next epoch

            curriculumChosenConsecutivelyTimes = self.calculateConsecutivelyChosen(curriculumChosenConsecutivelyTimes,
                                                                                   currentBestCurriculum,
                                                                                   lastChosenCurriculum,
                                                                                   self.curricula)
            lastChosenCurriculum = currentBestCurriculum

            self.updateTrainingInfo(epoch, iterationsDoneSoFar,
                                    self.curricula[currentBestCurriculum][startingIndex], currentBestCurriculum,
                                    rewards,
                                    self.getCurriculaEnvDetails(), curriculumChosenConsecutivelyTimes)
            # TODO : EVOLUTIONARY

        self.printFinalLogs()


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
