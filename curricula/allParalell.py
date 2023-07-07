import os


from curricula import train, evaluate, RollingHorizon
from utils import getEnvFromDifficulty, storage
from utils.curriculumHelper import *


class allParalell:
    def __init__(self, txtLogger, startTime, cmdLineString: str, args):
        # random.seed(args.seed)
        self.difficultyStepSize = args.difficultyStepsize
        self.args = args
        self.cmdLineString = cmdLineString
        self.lastEpochStartTime = startTime
        self.envDifficulty: float = 1.0
        self.seed = args.seed
        self.paraEnvs = len(ENV_NAMES.ALL_ENVS)

        self.ITERATIONS_PER_EVALUATE = args.iterPerEnv
        self.iterationsDone = 0
        self.txtLogger = txtLogger
        TOTAL_ITERATIONS = 10000000
        self.totalEpochs = TOTAL_ITERATIONS // self.ITERATIONS_PER_EVALUATE
        self.trainingTime = 0
        self.model = args.model + "_s" + str(self.seed)
        self.isSPLCL = not args.allSimultaneous

        self.selectedModel = self.model + os.sep + "model"

        self.stepMaxReward = calculateCurricStepMaxReward(ENV_NAMES.ALL_ENVS)

        self.trainingInfoJson = {}
        self.logFilePath = storage.getLogFilePath(["storage", self.model, "status.json"])

        self.curriculaEnvDetails = {}
        self.modelExists = os.path.exists(self.logFilePath)
        self.latestReward = 0
        self.startEpoch = 1

        if not self.isSPLCL:
            print("AllPara")
            self.initialEnvNames = self.updateEnvNamesNoAdjusment(self.envDifficulty)
        else:
            print("SPLCL")
            self.initialEnvNames = self.initializeEnvNames(self.envDifficulty) # todo just initialize with easiest
        if self.modelExists:
            self.loadTrainingInfo()
        else:
            self.trainingInfoJson = RollingHorizon.initTrainingInfo(self.cmdLineString, self.logFilePath, self.seed,
                                                                    self.stepMaxReward, None, self.args)

        self.trainEachCurriculum(self.startEpoch, self.totalEpochs, self.iterationsDone, self.initialEnvNames)

    def trainEachCurriculum(self, startEpoch: int, totalEpochs: int, iterationsDone: int, initialEnvNames: list):
        envNames = initialEnvNames
        print("training will go on until", totalEpochs)
        lastReward = 1
        for epoch in range(startEpoch, totalEpochs):
            self.txtLogger.info(f"Envs: {envNames }")
            iterationsDone = train.startTraining(iterationsDone + self.ITERATIONS_PER_EVALUATE, iterationsDone,
                                                 self.selectedModel, envNames, self.args, self.txtLogger)
            if epoch == 0:
                self.ITERATIONS_PER_EVALUATE = iterationsDone
                self.txtLogger.info(f"Exact iterations set: {iterationsDone} ")
            reward = evaluate.evaluateAgent(self.selectedModel, self.envDifficulty, self.args, self.txtLogger)
            self.envDifficulty = calculateEnvDifficulty(iterationsDone, self.difficultyStepSize)

            if not self.isSPLCL:
                envNames = self.updateEnvNamesNoAdjusment(self.envDifficulty)
            else:
                envNames = self.updateEnvNamesDynamically(envNames, self.envDifficulty, self.seed + epoch, reward)
            self.updateTrainingInfo(self.trainingInfoJson, epoch, envNames, reward, self.envDifficulty, iterationsDone)
            self.logInfoAfterEpoch(epoch, reward, self.txtLogger, totalEpochs)
            self.txtLogger.info(f"reward {reward}")

    @staticmethod
    def logInfoAfterEpoch(epoch, reward, txtLogger, totalEpochs):
        """
        Logs relevant training info after a training epoch is done and the trainingInfo was updated
        :param reward:
        :param txtLogger: the reference to the logger which logs all the info happening during training
        :param totalEpochs: the amount of epochs the training will go on for
        :param epoch: the current epoch nr
        :return:
        """
        currentEpoch = "epoch_" + str(epoch)
        txtLogger.info(f"Current rewards after {currentEpoch}: {reward}")
        txtLogger.info(f"\nEPOCH: {epoch} SUCCESS (total: {totalEpochs})\n ")

    @staticmethod
    def updateEnvNamesNoAdjusment(difficulty) -> list:
        envNames = []
        for j in range(len(ENV_NAMES.ALL_ENVS)):
            index = j
            envNames.append(getEnvFromDifficulty(index, difficulty))
        return envNames

    @staticmethod
    def initializeEnvNames(startDifficulty=1.0) -> list:
        envNames = []
        for j in range(len(ENV_NAMES.ALL_ENVS)):
            if j < 2:
                index = 0
            else:
                index = 1
            envNames.append(getEnvFromDifficulty(index, startDifficulty))
        return envNames

    @staticmethod
    def getHarderEnv(envName) -> str:
        index = ENV_NAMES.ALL_ENVS.index(envName) + 1
        if index >= len(ENV_NAMES.ALL_ENVS):
            index = len(ENV_NAMES.ALL_ENVS) - 1
        return ENV_NAMES.ALL_ENVS[index]

    @staticmethod
    def getEasierEnv(envName) -> str:
        index = ENV_NAMES.ALL_ENVS.index(envName) - 1
        if index < 0:
            index = 0
        return ENV_NAMES.ALL_ENVS[index]

    def updateEnvNamesDynamically(self, currentEnvNames: list, newDifficulty: float, seed: int, reward: float) -> list:
        envNames = currentEnvNames
        print("Before=", currentEnvNames, "newDiff", newDifficulty)
        # TODO get the value of the models progress (maybe last 3 runs, and then decide if you should go up or not)
        np.random.seed(seed)
        randomIndexSample = np.random.choice(range(len(ENV_NAMES.ALL_ENVS)), size=self.paraEnvs, replace=False)
        if reward > self.stepMaxReward * .75:
            nextStep = "goUp"
        elif reward > self.stepMaxReward * .5:
            nextStep = "stay"
        else:
            nextStep = "goDown"
        if nextStep == "stay":
            for i in range(len(envNames)):
                cutEnv = envNames[i].split("-custom")[0]
                envNames[i] = cutEnv + ENV_NAMES.CUSTOM_POSTFIX + str(newDifficulty)
        else:
            for i in randomIndexSample:
                cutEnv = envNames[i].split("-custom")[0]
                if nextStep == "goDown":
                    nextEnv = self.getEasierEnv(cutEnv)
                elif newDifficulty == "goUp":
                    nextEnv = self.getHarderEnv(cutEnv)
                else:
                    raise Exception("Invalid new difficulty")
                envNames[i] = nextEnv + ENV_NAMES.CUSTOM_POSTFIX + str(newDifficulty)
        for i in range(len(envNames)):
            cutEnv = envNames[i].split("-custom")[0]
            envNames[i] = cutEnv + ENV_NAMES.CUSTOM_POSTFIX + str(newDifficulty)
        print("env names after = ", envNames)
        return envNames

    def updateTrainingInfo(self, trainingInfoJson, epoch, envNames, reward, difficulty, framesDone):

        currentEpoch = "epoch_" + str(epoch)

        trainingInfoJson[epochsDone] = epoch + 1
        trainingInfoJson[numFrames] = framesDone
        trainingInfoJson[snapshotScoreKey].append(reward)

        trainingInfoJson[curriculaEnvDetailsKey][currentEpoch] = envNames
        trainingInfoJson[difficultyKey].append(difficulty)
        now = datetime.now()
        timeSinceLastEpoch = (now - self.lastEpochStartTime).total_seconds()
        trainingInfoJson[epochTrainingTime].append(timeSinceLastEpoch)
        trainingInfoJson[sumTrainingTime] += timeSinceLastEpoch
        trainingInfoJson["currentListOfCurricula"] = envNames

        saveTrainingInfoToFile(self.logFilePath, trainingInfoJson)
        self.lastEpochStartTime = now

    def loadTrainingInfo(self):
        with open(self.logFilePath, 'r') as f:
            self.trainingInfoJson = json.loads(f.read())

        if len(self.trainingInfoJson[difficultyKey]) > 0:
            self.envDifficulty = self.trainingInfoJson[difficultyKey][-1]
        self.iterationsDone = self.trainingInfoJson[numFrames]
        # self.initialEnvNames = self.trainingInfoJson["nextEnvList"]  # TODO test this
        self.startEpoch = self.trainingInfoJson[epochsDone]
        if len(self.trainingInfoJson[snapshotScoreKey]) > 0:
            self.latestReward = self.trainingInfoJson[snapshotScoreKey][-1]
        self.txtLogger.info(f'Reloading state\nContinuing from epoch {self.startEpoch}, seed: {self.seed}')
