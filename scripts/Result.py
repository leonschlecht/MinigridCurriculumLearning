import numpy as np

from curricula import RollingHorizonEvolutionaryAlgorithm
from utils.curriculumHelper import *


class Result:
    def __init__(self, evaluationDictionary):
        argsString: str = evaluationDictionary[fullArgs]
        self.loadedArgsDict: dict = {k.replace('Namespace(', ''): v for k, v in [pair.split('=') for pair in argsString.split(', ')]}
        self.modelName = self.loadedArgsDict[modelKey]
        self.selectedEnvList = evaluationDictionary[selectedEnvs]
        self.epochsTrained = evaluationDictionary[epochsDone] - 1
        self.framesTrained = evaluationDictionary[numFrames]
        self.modelPerformance = evaluationDictionary[
            actualPerformance]  # {"curricScoreRaw", "curricScoreNormalized", "snapshotScoreRaw", "curriculum"}
        self.bestCurriculaDict = evaluationDictionary[bestCurriculas]
        self.rewardsDict = evaluationDictionary[rewardsKey]
        self.stepMaxReward = evaluationDictionary[maxStepRewardKey]
        self.curricMaxReward = evaluationDictionary[maxCurricRewardKey]
        self.fullEnvDict = evaluationDictionary[curriculaEnvDetailsKey]
        self.difficultyList = evaluationDictionary[difficultyKey]
        self.trainingTimeList = evaluationDictionary[epochTrainingTime]
        self.trainingTimeSum = evaluationDictionary[sumTrainingTime]
        self.modelName = self.loadedArgsDict[argsModelKey]
        self.iterationsPerEnv = self.getIterationsPerEnv(evaluationDictionary, self.loadedArgsDict)

        self.epochDict = self.getEpochDict(self.rewardsDict)
        self.snapShotScores = self.getSnapshotScores(evaluationDictionary, self.modelPerformance)

        bestCurricScores = []
        avgEpochRewards = []
        numCurric = float(self.loadedArgsDict[numCurricKey])
        if self.loadedArgsDict[trainEvolutionary]: # TODO to method
            for epochKey in self.rewardsDict:
                epochDict = self.rewardsDict[epochKey]
                bestGen, bestIdx = RollingHorizonEvolutionaryAlgorithm.getGenAndIdxOfBestIndividual(epochDict)
                bestCurricScores.append(epochDict[GEN_PREFIX + bestGen][bestIdx])
                epochRewardsList = np.array(list(epochDict.values()))
                avgEpochRewards.append(np.sum(epochRewardsList))

            self.noOfGens: float = float(self.loadedArgsDict[nGenerations])
            self.maxCurricAvgReward = self.curricMaxReward * self.noOfGens * numCurric
        self.avgEpochRewards = avgEpochRewards
        self.bestCurricScore = bestCurricScores
        assert type(self.iterationsPerEnv) == int
        assert self.epochsTrained == len(self.rewardsDict.keys())

        assert usedEnvEnumerationKey in evaluationDictionary, f"UsedEnvs not found in Log File of model {self.modelName}"
        usedEnvEnumeration = evaluationDictionary[usedEnvEnumerationKey]
        self.snapshotEnvDistribution = self.getSnapshotEnvDistribution(self.selectedEnvList, usedEnvEnumeration)
        self.bestCurriculaEnvDistribution = self.getBestCurriculaEnvDistribution(self.bestCurriculaDict, usedEnvEnumeration)
        self.allCurricDistribution = self.getAllCurriculaEnvDistribution(self.fullEnvDict, usedEnvEnumeration)
        print("iterPerEnv", self.iterationsPerEnv, ";; iter done", self.iterationsPerEnv * self.epochsTrained)

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
        return [float(i) / self.stepMaxReward for i in snapshotScores]

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
            curriculum = self.getListOrDictEntry(bestCurriculaDict, epochList)
            for curricStep in curriculum:
                for env in curricStep:
                    envStrRaw = env.split("-custom")[0]
                    bestCurriculaEnvDistribution[envStrRaw] += 1
        return bestCurriculaEnvDistribution

    def getListOrDictEntry(self, selectedEnvList, entry):
        if type(selectedEnvList) == list:
            return entry
        else:
            return selectedEnvList[entry]

    def getSnapshotEnvDistribution(self, selectedEnvList, usedEnvEnumeration):
        snapshotEnvDistribution = {env: 0 for env in usedEnvEnumeration}
        for entry in selectedEnvList:
            curricSteps = self.getListOrDictEntry(selectedEnvList, entry)
            for env in curricSteps:
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
