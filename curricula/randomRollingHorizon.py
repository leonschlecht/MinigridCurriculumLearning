import numpy as np
import utils
from curricula import RollingHorizon


class RandomRollingHorizon(RollingHorizon):
    def __init__(self, txtLogger, startTime, cmdLineString: str, args, modelName):
        super().__init__(txtLogger, startTime, cmdLineString, args, modelName)

    def getCurrentBestCurriculum(self):
        currentBestCurriculumIdx = np.argmax(self.currentRewardsDict)
        currentBestCurriculum = self.curricula[currentBestCurriculumIdx]
        return currentBestCurriculum

    def getCurrentBestModel(self):
        currentBestCurriculumIdx = np.argmax(self.currentRewardsDict)
        currentBestModel = utils.getModelWithCurricSuffix(self.selectedModel, currentBestCurriculumIdx)
        return currentBestModel

    def updateSpecificInfo(self, epoch) -> None:
        # TODO this method should be renamed (this was intended for trainingInfoJson updates)
        self.curricula = self.randomlyInitializeCurricula(self.numCurric, self.stepsPerCurric, self.envDifficulty,
                                                          self.paraEnvs, self.allEnvs, self.seed + epoch)

    def executeOneEpoch(self, epoch: int) -> None:
        currentRewards = {"curric_" + str(i): [] for i in range(len(self.curricula))}
        snapshotRewards = {"curric_" + str(i): [] for i in range(len(self.curricula))}
        for i in range(len(self.curricula)):
            reward = self.trainACurriculum(i, self.iterationsDone, -1, self.curricula)
            currentRewards["curric_" + str(i)] = np.sum(reward)
            snapshotRewards["curric_" + str(i)] = reward[0]

        self.currentRewardsDict = currentRewards
        self.currentSnapshotRewards = snapshotRewards
        self.curriculaEnvDetails = self.curricula

    def getCurriculumName(self, i, genNr):
        assert genNr == -1, "this parameter shouldnt be set for RRH"
        return utils.getModelWithCurricSuffix(self.selectedModel, i)
