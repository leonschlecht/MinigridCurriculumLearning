import os
import numpy as np
import utils
from curricula import train, evaluate
from utils import getModelWithCandidatePrefix
from utils.curriculumHelper import *


class RandomRollingHorizon:
    """
    This class represents a "biased" random rolling horizon. I.e not fully random (like the class
    FullRandomHorizon.
    The difference is that the best curriculum is the base for the next iteration for x% of the curricula in the list
    This aims to achieve that given the last best curriculum, based on this we want to continue most of our training.
    """

    def __init__(self, txtLogger, startTime, cmdLineString: str, args, fullRandom, gamma=.9):
        assert args.envsPerCurric > 0
        assert args.numCurric > 0
        assert args.iterPerEnv > 0
        assert args.trainEpochs > 1  # TODO common file for the asserts here
        self.seed = args.seed
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
        self.cmdLineString = cmdLineString
        self.logFilePath = os.getcwd() + "\\storage\\" + self.args.model + "\\status.json"
        self.gamma = gamma
        self.selectedModel = utils.getEpochModelName(args.model, 0)  # TODO is this useful for 0?
        self.maxReward = calculateMaxReward(args.numCurric, gamma)

        self.envDifficulty = 0

        self.fullRandom = fullRandom

        self.startCurriculumTraining()

    def startCurriculumTraining(self) -> None:
        """
        Starts The RH Curriculum Training
        """
        iterationsDoneSoFar, startEpoch, lastChosenCurriculum, curricConsecutivelyChosen = self.initializeTrainingVariables(
            os.path.exists(self.logFilePath))
        fullRewardsDict = {}
        for epoch in range(startEpoch, self.totalEpochs):
            self.selectedModel = utils.getEpochModelName(self.model, epoch)
            currentRewards = {"curric_" + str(i): [] for i in range(len(self.curricula))}
            for i in range(len(self.curricula)):
                reward = self.trainEachCurriculum(i, iterationsDoneSoFar, epoch)
                currentRewards["curric_" + str(i)] = reward
            iterationsDoneSoFar += self.ITERATIONS_PER_ENV  # TODO use exact value; maybe return something during training
            currentBestCurriculumIdx = np.argmax(currentRewards)
            currentBestCurriculum = self.curricula[currentBestCurriculumIdx]
            utils.copyAgent(src=getModelWithCandidatePrefix(
                utils.getModelWithCurricSuffix(self.selectedModel, currentBestCurriculumIdx)),
                dest=utils.getEpochModelName(self.model, epoch + 1))  # the model for the next epoch

            fullRewardsDict["epoch_" + str(epoch)] = currentRewards
            currentScore = 0
            if not self.fullRandom:
                curricConsecutivelyChosen = \
                    self.calculateConsecutivelyChosen(curricConsecutivelyChosen, currentBestCurriculum,
                                                      lastChosenCurriculum)

            lastChosenCurriculum = currentBestCurriculum
            curriculaEnvDetails = self.curricula
            print("REWRADS=", fullRewardsDict)
            updateTrainingInfo(self.trainingInfoJson, epoch, currentBestCurriculum, fullRewardsDict, currentScore,
                               iterationsDoneSoFar, self.envDifficulty, self.lastEpochStartTime, self.curricula,
                               curriculaEnvDetails, self.logFilePath)
            self.lastEpochStartTime = datetime.now()
            if self.fullRandom:
                self.curricula = randomlyInitializeCurricula(self.numCurric, self.envsPerCurric, self.envDifficulty)
            else:
                self.curricula = self.updateCurriculaAfterHorizon(lastChosenCurriculum, self.numCurric,
                                                                  self.envDifficulty)
            saveTrainingInfoToFile(self.logFilePath, self.trainingInfoJson)
            self.envDifficulty = calculateEnvDifficulty(currentScore, self.maxReward)

        printFinalLogs(self.trainingInfoJson, self.txtLogger)

    def trainEachCurriculum(self, i: int, iterationsDone: int, epoch: int) -> int:
        """
        Simulates a horizon and returns the rewards obtained after evaluating the state at the end of the horizon
        """
        nameOfCurriculumI = utils.getModelWithCurricSuffix(self.selectedModel, i)  # Save TEST_e1 --> TEST_e1_curric0
        rewards = 0  # TODO epoch param in method here ???
        utils.copyAgent(src=self.selectedModel, dest=nameOfCurriculumI)
        for j in range(len(self.curricula[i])):
            iterationsDone = train.main(iterationsDone + self.ITERATIONS_PER_ENV, iterationsDone, nameOfCurriculumI,
                                        self.curricula[i][j], self.args, self.txtLogger)
            self.txtLogger.info(f"Iterations Done {iterationsDone}")
            if j == 0:
                utils.copyAgent(src=nameOfCurriculumI, dest=utils.getModelWithCandidatePrefix(
                    nameOfCurriculumI))  # save TEST_e1_curric0 -> + _CANDIDATE
                # TODO save reward separately here?
            self.txtLogger.info(f"Trained iteration j={j} of curriculum {nameOfCurriculumI} ")
            rewards += ((self.gamma ** j) * evaluate.evaluateAgent(nameOfCurriculumI, self.envDifficulty, self.args))

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
            self.lastEpochStartTime = self.trainingInfoJson["startTime"]  # TODO use right keys
            consec = self.trainingInfoJson[consecutivelyChosen]  # TODO load
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

            self.trainingInfoJson = initTrainingInfo(self.cmdLineString, self.logFilePath, self.seed, self.args)
            utils.copyAgent(src=self.selectedModel, dest=utils.getEpochModelName(self.model, 1))
            lastChosenCurriculum = None
            consec = 0
        return iterationsDoneSoFar, startEpoch, lastChosenCurriculum, consec

    def calculateConsecutivelyChosen(self, consecutiveCount, currentBestCurriculum, lastChosenCurriculum) -> int:
        if consecutiveCount + 1 < len(self.curricula[0]) and \
                (currentBestCurriculum == lastChosenCurriculum or lastChosenCurriculum is None):
            return consecutiveCount + 1
        return 0

    def updateCurriculaAfterHorizon(self, bestCurriculum: list, numberOfCurricula: int, envDifficulty: int) -> list:
        """
        Updates the List of curricula by using the last N-1 Envs, and randomly selecting a last new one
        :param envDifficulty:
        :param numberOfCurricula:
        :param bestCurriculum: full env list of the curriculum that performed best during last epoch
                (i.e. needs to be cut by 1 element!)
        """
        curricula = []
        # TODO remove duplicates
        # TODO dealing with the case where duplicates are forced due to parameters
        # TODO maybe only do this for a percentage of curricula, and randomly set the others OR instead of using [1:], use [1:__]
        # TODO test this
        for i in range(numberOfCurricula):
            curricula.append(bestCurriculum[1:])
            envId = np.random.randint(0, len(ENV_NAMES.ALL_ENVS))
            curricula[i].append(getEnvFromDifficulty(envId, envDifficulty))
        assert len(curricula) == numberOfCurricula
        return curricula
