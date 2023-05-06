import os
import numpy as np
import utils
from curricula import train
from utils import getModelWithCandidatePrefix
from utils.curriculumHelper import *


class FullRandomRollingHorizon:
    """
    This class represents a "biased" random rolling horizon. I.e not fully random (like the class
    FullRandomHorizon.
    The difference is that the best curriculum is the base for the next iteration for x% of the curricula in the list
    This aims to achieve that given the last best curriculum, based on this we want to continue most of our training.
    """

    def __init__(self, txtLogger, startTime, args, gamma=.9):
        assert args.envsPerCurric > 0
        assert args.numCurric > 0
        assert args.iterPerEnv > 0
        assert args.trainEpochs > 1  # TODO common file for the asserts here

        self.numCurric = args.numCurric
        self.totalEpochs = args.trainEpochs
        self.envsPerCurric = args.numCurric
        self.ITERATIONS_PER_ENV = args.iterPerEnv
        self.txtLogger = txtLogger
        self.args = args
        self.lastEpochStartTime = startTime
        self.trainingInfoJson = {}
        self.curricula = []
        self.model = args.model
        self.logFilePath = os.getcwd() + "\\storage\\" + self.args.model + "\\status.json"
        self.gamma = gamma
        self.selectedModel = utils.getEpochModelName(args.model, 0)  # TODO is this useful for 0?

        self.envDifficulty = 0

        self.startCurriculumTraining()

    def startCurriculumTraining(self) -> None:
        """
        Starts The RH Curriculum Training
        """
        iterationsDoneSoFar, startEpoch, lastChosenCurriculum = self.initializeTrainingVariables(
            os.path.exists(self.logFilePath))
        envDifficulty = self.envDifficulty
        for epoch in range(startEpoch, 11):
            self.selectedModel = utils.getEpochModelName(self.model, epoch)
            rewards = {str(i): [] for i in range(len(self.curricula))}
            for i in range(len(self.curricula)):
                reward = self.trainEachCurriculum(i, iterationsDoneSoFar, self.selectedModel, epoch)
                rewards[str(i)].append(reward)
            iterationsDoneSoFar += self.ITERATIONS_PER_ENV  # TODO use exact value; maybe return something during training
            currentBestCurriculumIdx = int(
                np.argmax([lst[-1] for lst in rewards.values()]))  # only access the latest reward
            currentBestCurriculum = self.curricula[currentBestCurriculumIdx]
            utils.copyAgent(src=getModelWithCandidatePrefix(
                utils.getModelWithCurricSuffix(self.selectedModel, epoch, currentBestCurriculumIdx)),
                            dest=utils.getEpochModelName(self.model, epoch + 1))  # the model for the next epoch
            currentRewards = 0
            currentScore = 0
            print("REWRADS=", rewards)
            updateTrainingInfo(self.trainingInfoJson, epoch, currentBestCurriculum, currentRewards, currentScore,
                               iterationsDoneSoFar, envDifficulty, self.lastEpochStartTime, self.curricula,
                               self.logFilePath)
            self.lastEpochStartTime = datetime.now()
            self.curricula = randomlyInitializeCurricula(self.numCurric, self.envsPerCurric,
                                                         envDifficulty)
            self.trainingInfoJson["currentCurriculumList"] = self.curricula  # TODO refactor this
            saveTrainingInfoToFile(self.logFilePath, self.trainingInfoJson)
            envDifficulty = 0
            print("---- EOCH END ---- \n")
            # TODO update env difficulty

        printFinalLogs(self.trainingInfoJson, self.txtLogger)

    def trainEachCurriculum(self, i: int, iterationsDone: int, selectedModel: str, epoch: int) -> int:
        """
        Simulates a horizon and returns the rewards obtained after evaluating the state at the end of the horizon
        """
        nameOfCurriculumI = utils.getModelWithCurricSuffix(selectedModel, epoch, i)  # Save TEST_e1 --> TEST_e1_curric0
        rewards = 0
        utils.copyAgent(src=selectedModel, dest=nameOfCurriculumI)
        for j in range(len(self.curricula[i])):
            iterationsDone = train.main(iterationsDone + self.ITERATIONS_PER_ENV, iterationsDone, nameOfCurriculumI,
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
            self.lastEpochStartTime = self.trainingInfoJson["startTime"]
            # delete existing folders, that were created ---> maybe just last one because others should be finished ...
            for k in range(self.numCurric):
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
            self.curricula = randomlyInitializeCurricula(self.numCurric, self.envsPerCurric,
                                                         self.envDifficulty)
            iterationsDoneSoFar = train.main(0, 0, self.selectedModel, getEnvFromDifficulty(0, self.envDifficulty),
                                             self.args, self.txtLogger)

            self.trainingInfoJson = initTrainingInfo("TODO", self.logFilePath)  # TODO cmdLineSTring
            utils.copyAgent(src=self.selectedModel, dest=utils.getEpochModelName(self.model, 1))
            lastChosenCurriculum = None
        return iterationsDoneSoFar, startEpoch, lastChosenCurriculum
