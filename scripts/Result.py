import numpy as np

from curricula import RollingHorizonEvolutionaryAlgorithm
from utils.curriculumHelper import *


class Result:
    def __init__(self, evaluationDictionary):
        argsString: str = evaluationDictionary[fullArgs]
        self.loadedArgsDict: dict = {k.replace('Namespace(', ''): v for k, v in [pair.split('=') for pair in argsString.split(', ')]}

        self.selectedEnvList = evaluationDictionary[selectedEnvs]
        self.epochsTrained = evaluationDictionary[epochsDone] - 1
        self.framesTrained = evaluationDictionary[numFrames]
        self.modelPerformance = evaluationDictionary[
            actualPerformance]  # {"curricScoreRaw", "curricScoreNormalized", "snapshotScoreRaw", "curriculum"}
        self.snapShotScores = self.getSnapshotScores(evaluationDictionary, self.modelPerformance)
        self.bestCurriculaDict = evaluationDictionary[bestCurriculas]
        self.rewardsDict = evaluationDictionary[rewardsKey]

        self.stepMaxReward = evaluationDictionary[maxStepRewardKey]
        self.curricMaxReward = evaluationDictionary[maxCurricRewardKey]

        self.fullEnvDict = evaluationDictionary[curriculaEnvDetailsKey]
        self.difficultyList = evaluationDictionary[difficultyKey]
        self.trainingTimeList = evaluationDictionary[epochTrainingTime]
        self.trainingTimeSum = evaluationDictionary[sumTrainingTime]

        self.epochDict = self.getEpochDict(self.rewardsDict)

        self.modelName = self.loadedArgsDict[argsModelKey]
        self.iterationsPerEnv = self.getIterationsPerEnv(evaluationDictionary, self.loadedArgsDict)

        curricScores = []
        avgEpochRewards = []
        numCurric = float(self.loadedArgsDict[numCurricKey])
        if self.loadedArgsDict[trainEvolutionary]:
            for epochKey in self.rewardsDict:
                epochDict = self.rewardsDict[epochKey]
                genNr, listIdx = RollingHorizonEvolutionaryAlgorithm.getGenAndIdxOfBestIndividual(epochDict)
                bestCurricScore = epochDict[GEN_PREFIX + genNr][listIdx]
                curricScores.append(bestCurricScore)
                epochRewardsList = np.array(list(epochDict.values()))
                avgEpochRewards.append(np.sum(epochRewardsList))

            self.noOfGens: float = float(self.loadedArgsDict[nGenerations])
            self.maxCurricAvgReward = self.curricMaxReward * self.noOfGens * numCurric

        assert type(self.iterationsPerEnv) == int
        assert self.epochsTrained == len(self.rewardsDict.keys())

        usedEnvEnumeration = evaluationDictionary[usedEnvEnumerationKey]
        snapshotEnvDistribution = self.getSnapshotEnvDistribution(self.selectedEnvList, usedEnvEnumeration)
        bestCurriculaEnvDistribution = self.getBestCurriculaEnvDistribution(self.bestCurriculaDict, usedEnvEnumeration)
        allCurricDistribution = self.getAllCurriculaEnvDistribution(self.fullEnvDict, usedEnvEnumeration)

        # plotEnvsUsedDistribution(allCurricDistribution, "all Curric Distribution")
        # plotEnvsUsedDistribution(snapshotEnvDistribution, "snapshot Distribution")
        # plotEnvsUsedDistribution(bestCurriculaEnvDistribution, "best Curricula Distribution")
        # plotSnapshotPerformance(snapshotScores, stepMaxReward, modelName, iterationsPerEnv)
        # plotBestCurriculumResults(curricScores, curricMaxReward, modelName, iterationsPerEnv)
        # plotEpochAvgCurricReward(avgEpochRewards, maxCurricAvgReward, modelName, iterationsPerEnv)

        # TODO plot the snapshot vs curricReward problem
        # TODO plot reward development of 1 curriculum over multiple generations
        # TODO find out a way to properly plot the difficulty list / maybe how it influences the results; and maybe how you can improve it so that it is not even needed in the first place
        # TODO find way to plot multiple models at once (and show some relevant legend for info of model name or sth like that)
        # TODO save the plots

    def getSnapshotScores(self, evalDict: dict, modelPerformance: dict) -> list[float]:
        """
        Gets the 1st step scores of the model and returns it as a list.
        :param evalDict:
        :param modelPerformance:
        :return:
        """
        if snapshotScoreKey in evalDict.keys():
            snapshotScores = evalDict[snapshotScoreKey]
        else:
            snapshotScores = []
            for epochDict in modelPerformance:
                snapshotScores.append(epochDict[snapshotScoreKey])
        return snapshotScores

    def getEpochDict(self, rewardsDict):
        """
        Transforms the str list of the rewards dictionary into a dictionary that has the lists with float values instead of str values
        """
        epochDict = {}
        for epochKey in rewardsDict:
            epochDict = rewardsDict[epochKey]
            for genKey in epochDict:
                genRewardsStr = epochDict[genKey]
                numbers = re.findall(r'[0-9]*\.?[0-9]+', genRewardsStr)
                numbers = [float(n) for n in numbers]
                epochDict[genKey] = numbers
            # TODO assertion to make sure length hasnt chagned
        return epochDict

    def getBestCurriculaEnvDistribution(self, bestCurriculaDict, usedEnvEnumeration):
        bestCurriculaEnvDistribution = {env: 0 for env in usedEnvEnumeration}
        for epochList in bestCurriculaDict:
            for curricStep in epochList:
                for env in curricStep:  # TODO this might break because of recent log changes -> test it
                    envStrRaw = env.split("-custom")[0]
                    bestCurriculaEnvDistribution[envStrRaw] += 1
        return bestCurriculaEnvDistribution

    def getSnapshotEnvDistribution(self, selectedEnvList, usedEnvEnumeration):
        snapshotEnvDistribution = {env: 0 for env in usedEnvEnumeration}
        for curricStep in selectedEnvList:
            for env in curricStep:
                envStrRaw = env.split("-custom")[0]
                snapshotEnvDistribution[envStrRaw] += 1
        return snapshotEnvDistribution

    def getAllCurriculaEnvDistribution(self, fullEnvDict, usedEnvEnumeration):
        allCurricDistribution = {env: 0 for env in usedEnvEnumeration}
        for epochKey in fullEnvDict:
            epochGenDict = fullEnvDict[epochKey]
            for genKey in epochGenDict:
                curriculumList = epochGenDict[genKey]
                for curric in curriculumList:
                    for curricStep in curric:
                        for env in curricStep:
                            envStrRaw = env.split("-custom")[0]
                            allCurricDistribution[envStrRaw] += 1
        return allCurricDistribution

    def getIterationsPerEnv(self, trainingInfoDict, loadedArgsDict):
        if iterationsPerEnvKey in trainingInfoDict.keys():
            iterationsPerEnv = int(trainingInfoDict[iterationsPerEnvKey])
        else:
            iterationsPerEnv = int(loadedArgsDict[oldArgsIterPerEnvName])  # TODO this might become deprecated if I change iterPerEnv -> stepsPerEnv

        return iterationsPerEnv
