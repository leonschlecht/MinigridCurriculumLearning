import numpy as np
from matplotlib import pyplot as plt, cm

from curricula import RollingHorizonEvolutionaryAlgorithm
from utils.curriculumHelper import *


def getTransformedReward(evaluationDictionary):
    """
    Helper function that transforms the string of the json containing all the rewards (as strings!) to an actual dict with lists
    :return:
    """
    rawRewardStr = evaluationDictionary["rawRewards"]
    transformedRawReward = {}
    for epoch in rawRewardStr:
        transformedRawReward[epoch] = {}
        for gen in rawRewardStr[epoch]:
            transformedRawReward[epoch][gen] = {}
            for curric in rawRewardStr[epoch][gen]:
                transformedRawReward[epoch][gen][curric] = []
                s = (rawRewardStr[epoch][gen][curric])[1:-1]
                tmp = s.split("\n")
                for curricStep in tmp:
                    numbers = re.findall(r'[0-9]*\.?[0-9]+', curricStep)
                    numbers = [float(n) for n in numbers]
                    transformedRawReward[epoch][gen][curric].append(numbers)
    return transformedRawReward

class Result:
    def __init__(self, evaluationDictionary, modelName, logfilePath):
        # NOTE: the modelName is the name of the directory; not the name of the --model command when the training was performed
        # This is due to naming convenience / overview in the evaluation
        argsString: str = evaluationDictionary[fullArgs]
        self.canceled = modelName[0] == "C" and modelName[1] == "_"
        self.loadedArgsDict: dict = {k.replace('Namespace(', ''): v for k, v in
                                     [pair.split('=') for pair in argsString.split(', ')]}
        self.loadedArgsDict[trainEvolutionary] = self.getTrainEvolutionary(self.loadedArgsDict[trainEvolutionary])
        self.modelName = self.loadedArgsDict[modelKey]
        self.seed = self.loadedArgsDict[seedKey]
        self.selectedEnvDict = evaluationDictionary[selectedEnvs]  # TODO fix this for all para and others maybe ??
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
        self.trainingTimeSum = evaluationDictionary[sumTrainingTime] / 60 / 60
        self.modelName = modelName
        self.iterationsPerEnv = self.getIterationsPerEnv(evaluationDictionary, self.loadedArgsDict)
        self.iterationsPerEnv = int(self.loadedArgsDict["iterPerEnv"])
        self.logFilepath = logfilePath

        self.snapShotScores = self.getSnapshotScores(evaluationDictionary, self.modelPerformance)
        self.iterationsList = []
        for i in range(1, len(self.snapShotScores) + 1):
            self.iterationsList.append(self.iterationsPerEnv * i)

        bestCurricScores = []
        avgEpochRewards = []
        numCurric = float(self.loadedArgsDict[numCurricKey])
        usedEnvEnumeration = evaluationDictionary[usedEnvEnumerationKey]

        if self.loadedArgsDict[trainEvolutionary]:  # TODO move to method
            self.epochDict = self.getEpochDict(self.rewardsDict)
            self.noOfGens: float = float(self.loadedArgsDict[nGenerations])
            self.maxCurricAvgReward = self.curricMaxReward * self.noOfGens * numCurric
            for epochKey in self.rewardsDict:
                epochDict = self.rewardsDict[epochKey]
                bestGen, bestIdx = RollingHorizonEvolutionaryAlgorithm.getGenAndIdxOfBestIndividual(epochDict)
                bestCurricScores.append(epochDict[GEN_PREFIX + bestGen][bestIdx] / self.curricMaxReward)
                epochRewardsList = np.array(list(epochDict.values()))
                avgEpochRewards.append(np.sum(epochRewardsList) / self.maxCurricAvgReward)
            self.allCurricDistribution = self.getAllCurriculaEnvDistribution(self.fullEnvDict, usedEnvEnumeration)
            self.snapshotEnvDistribution = self.getSnapshotEnvDistribution(self.selectedEnvDict, usedEnvEnumeration)
            self.bestCurriculaEnvDistribution, self.splitDistrBestCurric = \
                self.getBestCurriculaEnvDistribution(self.bestCurriculaDict, usedEnvEnumeration)
            formatHelperList = [[env for env in self.selectedEnvDict[epoch]] for epoch in self.selectedEnvDict]
            # keep only the env size part, e.g. 6x6
            self.formattedSelectedEnvList = [[env.split("-")[2] for env in sublist] for sublist in formatHelperList]

            assert len(self.formattedSelectedEnvList) == len(self.snapShotScores), \
                "Something went went wrong with creating the formatted selected env list"
            if "newLog" in self.modelName:
                self.rawReward = getTransformedReward(evaluationDictionary)

            if not self.canceled:

                """
                x = self.iterationsList
                y = self.formattedSelectedEnvList
                y1 = list(zip(*y))[0]
                y2 = list(zip(*y))[1]

                y_combined = y1 + y2
                unique_y = np.unique(y_combined)
                num_unique_y = 4
                assert len(unique_y) == 4, f"There should be 4 envs not, {unique_y}"
                cmap = cm.get_cmap("Set3", num_unique_y)  # Get the Set3 colormap with the number of unique y-values

                colors = [cmap(i) for i in range(num_unique_y)]
                color_mapping = dict(zip(unique_y, colors))
                plt.scatter(x, y1, color=[color_mapping[y] for y in y1], s=5)
                plt.scatter(x, y2, color=[color_mapping[y] for y in y2], s=5)

                #unique_envs = list(set([env for sublist in y for env in sublist]))
                #unique_envs.sort(key=lambda env: int(env.split('x')[0]), reverse=False)
                # plt.yticks(range(len(unique_envs)), unique_envs)
                plt.show()
                """


        elif self.loadedArgsDict[trainRandomRH] == "True":
            self.allCurricDistribution = {env: 0 for env in usedEnvEnumeration}
            self.bestCurriculaEnvDistribution = {env: 0 for env in usedEnvEnumeration}
            self.snapshotEnvDistribution = {env: 0 for env in usedEnvEnumeration}

            for helper in self.fullEnvDict:
                for tmp in self.fullEnvDict[helper]:
                    for t in tmp:
                        for env in t:
                            envStrRaw = env.split("-custom")[0]  # TODO this is duplicate in other method
                            self.allCurricDistribution[envStrRaw] += 1
            for epochKey in self.bestCurriculaDict:
                i = 0
                for stepOfBestCurric in self.bestCurriculaDict[epochKey]:
                    for env in stepOfBestCurric:
                        envStrRaw = env.split("-custom")[0]  # TODO this is duplicate in other method
                        self.bestCurriculaEnvDistribution[envStrRaw] += 1
                        if i == 0:
                            self.snapshotEnvDistribution[envStrRaw] += 1
                    i += 1
            # TODO move to new method
            bestCurricScores = self.snapShotScores
            avgEpochRewards = self.snapShotScores
        elif self.loadedArgsDict[trainAllParalell]:
            self.allCurricDistribution = []
            bestCurricScores = self.snapShotScores
            avgEpochRewards = self.snapShotScores
            self.snapshotEnvDistribution = {env: 0 for env in usedEnvEnumeration}
            if self.loadedArgsDict["allSimultaneous"] == "True":
                for env in usedEnvEnumeration:
                    self.snapshotEnvDistribution[env] = self.epochsTrained
            else:
                for env in usedEnvEnumeration:
                    self.snapshotEnvDistribution[env] = 0
                for helper in self.fullEnvDict:
                    for epochResult in self.fullEnvDict[helper]:
                        envStrRaw = epochResult.split("-custom")[0]  # TODO this is duplicate in other method
                        self.snapshotEnvDistribution[envStrRaw] += 1
            self.bestCurriculaEnvDistribution = self.snapshotEnvDistribution
            self.allCurricDistribution = self.snapshotEnvDistribution
            # TODO move all this code out of if else maybe ?

        self.avgEpochRewards = avgEpochRewards
        self.bestCurricScore = bestCurricScores

        errorPrefix = f"model: {self.modelName}_s{self.loadedArgsDict[seedKey]}:"
        assert self.bestCurricScore != []
        assert self.avgEpochRewards != []
        assert len(self.iterationsList) == len(self.snapShotScores), \
            f"{errorPrefix} {len(self.iterationsList)} and {len(self.snapShotScores)} "
        assert type(self.iterationsPerEnv) == int
        assert self.epochsTrained == len(self.rewardsDict.keys()) or not self.loadedArgsDict[trainEvolutionary]
        assert usedEnvEnumerationKey in evaluationDictionary, f"UsedEnvs not found in Log File of model {self.logFilepath}"
        assert sum(self.snapshotEnvDistribution.values()) > 0
        assert sum(self.bestCurriculaEnvDistribution.values()) > 0
        assert len(self.snapShotScores) == len(self.bestCurricScore) == len(self.avgEpochRewards), \
            f"{errorPrefix} {len(self.snapShotScores)}, {len(self.bestCurricScore)}, {len(self.avgEpochRewards)}"

    @staticmethod
    def getTrainEvolutionary(param):
        # this is because of bad conversion from an args object that is stored in a txt file
        if param == "True)":
            return True
        elif param == "False)":
            return False
        else:
            raise Exception("Error while parsing train evolutionary parameter")

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
        splitDistributions = []
        if self.iterationsList[-1] * self.iterationsPerEnv > 5000000 and self.loadedArgsDict[trainEvolutionary]:
            i = 0
            splitDistributions = []
            bestCurriculaEnvDistribution = {env: 0 for env in usedEnvEnumeration}
            for epochList in bestCurriculaDict:
                curriculum = self.getListOrDictEntry(bestCurriculaDict, epochList)
                epochNr = int(epochList.split("_")[1])
                if epochNr * self.iterationsPerEnv > 1000000 * (i + 1):  # TODO maybe here for the 500k steps
                    splitDistributions.append(bestCurriculaEnvDistribution)
                    bestCurriculaEnvDistribution = {env: 0 for env in usedEnvEnumeration}
                    i += 1
                for curricStep in curriculum:
                    for env in curricStep:
                        envStrRaw = env.split("-custom")[0]
                        bestCurriculaEnvDistribution[envStrRaw] += 1
            splitDistributions.append(bestCurriculaEnvDistribution)

        bestCurriculaEnvDistribution = {env: 0 for env in usedEnvEnumeration}
        for epochList in bestCurriculaDict:
            curriculum = self.getListOrDictEntry(bestCurriculaDict, epochList)
            for curricStep in curriculum:
                for env in curricStep:
                    envStrRaw = env.split("-custom")[0]
                    bestCurriculaEnvDistribution[envStrRaw] += 1
        if splitDistributions is not []:
            splitDistrSum = 0
            for split in splitDistributions:
                splitDistrSum += sum(split.values())
            assert sum(bestCurriculaEnvDistribution.values()) == splitDistrSum, "Splitting the env distribution went wrong"
        return bestCurriculaEnvDistribution, splitDistributions

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
            iterationsPerEnv = int(loadedArgsDict[
                                       oldArgsIterPerEnvName])  # TODO this might become deprecated if I change iterPerEnv -> stepsPerEnv

        return iterationsPerEnv

    def finishAggregation(self, snapshotScores, bestCurricScores, avgEpochRewards, snapshotDistr, avgBestCurricDistr,
                          avgAllCurricDistr, amountOfTrainingRuns: int) -> None:
        # # TODO (Maybe aggregate difficulty too)
        # TODO is this still needed ?
        assert len(self.snapShotScores) == len(self.bestCurricScore) == len(self.avgEpochRewards)

        self.snapShotScores = np.divide(snapshotScores, amountOfTrainingRuns)
        self.bestCurricScore = np.divide(bestCurricScores, amountOfTrainingRuns)
        self.avgEpochRewards = np.divide(avgEpochRewards, amountOfTrainingRuns)

        self.snapshotEnvDistribution = snapshotDistr
        self.bestCurriculaEnvDistribution = avgBestCurricDistr
        self.allCurricDistribution = avgAllCurricDistr  # TODO is this useful to aggegrate?

        for k in self.snapshotEnvDistribution.keys():
            self.snapshotEnvDistribution[k] /= amountOfTrainingRuns
            self.bestCurriculaEnvDistribution[k] /= amountOfTrainingRuns
            self.allCurricDistribution[k] /= amountOfTrainingRuns
