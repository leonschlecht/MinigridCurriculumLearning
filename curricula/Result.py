from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm

from curricula import RollingHorizonEvolutionaryAlgorithm
from utils.curriculumHelper import *
import seaborn as sns


def getTransformedReward(evaluationDictionary, isRHEA, isRRH):
    """
    Helper function that transforms the string of the json containing all the rewards (as strings!) to an actual dict with lists
    :return:
    """
    rawRewardStr = evaluationDictionary["rawRewards"]
    transformedRawReward = {}
    if isRHEA:
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
    elif isRRH:
        for epoch in rawRewardStr:
            transformedRawReward[epoch] = {}
            for curric in rawRewardStr[epoch]:
                transformedRawReward[epoch][curric] = {}
                transformedRawReward[epoch][curric] = []
                s = (rawRewardStr[epoch][curric])[1:-1]
                tmp = s.split("\n")
                for curricStep in tmp:
                    numbers = re.findall(r'[0-9]*\.?[0-9]+', curricStep)
                    numbers = [float(n) for n in numbers]
                    transformedRawReward[epoch][curric].append(numbers)
    else:
        return rawRewardStr
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
        curricLen = int(self.loadedArgsDict["stepsPerCurric"])
        self.gamma = float(self.loadedArgsDict["gamma"])
        if self.loadedArgsDict[trainEvolutionary]:  # TODO move to method
            self.epochDict = self.getEpochDict(self.rewardsDict)
            self.noOfGens: float = float(self.loadedArgsDict[nGenerations])
            self.maxCurricAvgReward = self.curricMaxReward * self.noOfGens * numCurric  # WTH is this ??
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

            if not self.canceled:
                """
                # TODO selected env plot
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
                for bestGenDict in self.fullEnvDict[helper]:
                    for t in bestGenDict:
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
        self.usedEnvEnumeration = usedEnvEnumeration

        if "rawRewards" in evaluationDictionary.keys():
            self.rawReward = getTransformedReward(evaluationDictionary, self.loadedArgsDict[trainEvolutionary],
                                                  self.loadedArgsDict[trainRandomRH] == "True")
            env1, env2, env3, env4, bestGenDict = self.getEnvRewards(self.loadedArgsDict[trainEvolutionary],
                                                                     self.loadedArgsDict[trainRandomRH] == "True",
                                                                     curricLen)
            self.env1 = env1
            self.env2 = env2
            self.env3 = env3
            self.env4 = env4
            self.bestGenDict = bestGenDict
        else:
            self.rawReward = None
            self.env1 = None
            self.env2 = None
            self.env3 = None
            self.env4 = None

        errorPrefix = f"model: {self.modelName}_s{self.loadedArgsDict[seedKey]}:"
        assert self.bestCurricScore != []
        assert self.avgEpochRewards != []
        assert len(self.iterationsList) == len(self.snapShotScores), \
            f"{errorPrefix} {len(self.iterationsList)} and {len(self.snapShotScores)} "
        assert type(self.iterationsPerEnv) == int
        assert self.epochsTrained == len(self.rewardsDict.keys()) or not self.loadedArgsDict[trainEvolutionary]
        assert not np.isnan(self.epochsTrained), f"Nan for {self.modelName} in epochs trained"
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
        return [round(float(i) / self.stepMaxReward, 4) for i in snapshotScores]

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
            assert sum(
                bestCurriculaEnvDistribution.values()) == splitDistrSum, "Splitting the env distribution went wrong"
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
            iterationsPerEnv = int(loadedArgsDict[oldArgsIterPerEnvName])
        return iterationsPerEnv

    def getScoreAtStepI(self, i):
        assert self.snapShotScores[i] < 1
        if self.env1 is not None:
            envScores = {"env1": self.env1[i],
                         "env2": self.env2[i],
                         "env3": self.env3[i],
                         "env4": self.env4[i]}
        else:
            envScores = {}
        return ({"snapshotScore": self.snapShotScores[i],
                 "bestCurricScore": self.bestCurricScore[i],
                 "avgEpochRewards": self.avgEpochRewards[i],
                 **envScores,
                 "id": self.modelName,
                 "seed": self.seed,
                 "group": self.iterationsPerEnv,  # TODO refactor this (column name)
                 iterationSteps: self.iterationsList[i]})

    def getDistributions(self, isDoorKey):
        # Create DF2 that contains the distributions etc. (iterationNr column does not make sense here)

        keys = self.snapshotEnvDistribution.keys()
        snapshotHelper = [[] for _ in keys]
        allCurricHelper = [[] for _ in keys]
        bestCurricHelper = [[] for _ in keys]
        for k in keys:
            if isDoorKey:
                if "6x6" in k:
                    snapshotHelper[0] = (self.snapshotEnvDistribution[k])
                    allCurricHelper[0] = (self.allCurricDistribution[k])
                    bestCurricHelper[0] = (self.bestCurriculaEnvDistribution[k])
                elif "8x8" in k:
                    snapshotHelper[1] = (self.snapshotEnvDistribution[k])
                    allCurricHelper[1] = (self.allCurricDistribution[k])
                    bestCurricHelper[1] = (self.bestCurriculaEnvDistribution[k])
                elif "10x10" in k:
                    snapshotHelper[2] = (self.snapshotEnvDistribution[k])
                    allCurricHelper[2] = (self.allCurricDistribution[k])
                    bestCurricHelper[2] = (self.bestCurriculaEnvDistribution[k])
                else:
                    snapshotHelper[3] = (self.snapshotEnvDistribution[k])
                    allCurricHelper[3] = (self.allCurricDistribution[k])
                    bestCurricHelper[3] = (self.bestCurriculaEnvDistribution[k])
            else:
                if "5x5" in k:
                    snapshotHelper[0] = (self.snapshotEnvDistribution[k])
                    allCurricHelper[0] = (self.allCurricDistribution[k])
                    bestCurricHelper[0] = (self.bestCurriculaEnvDistribution[k])
                elif "6x6" in k:
                    snapshotHelper[1] = (self.snapshotEnvDistribution[k])
                    allCurricHelper[1] = (self.allCurricDistribution[k])
                    bestCurricHelper[1] = (self.bestCurriculaEnvDistribution[k])
                elif "8x8" in k:
                    snapshotHelper[2] = (self.snapshotEnvDistribution[k])
                    allCurricHelper[2] = (self.allCurricDistribution[k])
                    bestCurricHelper[2] = (self.bestCurriculaEnvDistribution[k])
                else:  # 16x16
                    snapshotHelper[3] = (self.snapshotEnvDistribution[k])
                    allCurricHelper[3] = (self.allCurricDistribution[k])
                    bestCurricHelper[3] = (self.bestCurriculaEnvDistribution[k])
        if isDoorKey:
            return {
                "6x6s": snapshotHelper[0],
                "8x8s": snapshotHelper[1],
                "10x10s": snapshotHelper[2],
                "12x12s": snapshotHelper[3],
                "6x6c": bestCurricHelper[0],
                "8x8c": bestCurricHelper[1],
                "10x10c": bestCurricHelper[2],
                "12x12c": bestCurricHelper[3],
                "6x6a": allCurricHelper[0],
                "8x8a": allCurricHelper[1],
                "10x10a": allCurricHelper[2],
                "12x12a": allCurricHelper[3],
                seedKey: self.seed,
                "group": self.iterationsPerEnv,
                iterationSteps: self.iterationsList[0],
                sumTrainingTime: self.trainingTimeSum,
                "id": self.modelName}
        else:
            return {
                "5x5s": snapshotHelper[0],
                "6x6s": snapshotHelper[1],
                "8x8s": snapshotHelper[2],
                "16x16s": snapshotHelper[3],
                "5x5c": bestCurricHelper[0],
                "6x6c": bestCurricHelper[1],
                "8x8c": bestCurricHelper[2],
                "16x16c": bestCurricHelper[3],
                "5x5a": allCurricHelper[0],
                "6x6a": allCurricHelper[1],
                "8x8a": allCurricHelper[2],
                "16x16a": allCurricHelper[3],
                seedKey: self.seed,
                "group": self.iterationsPerEnv,
                iterationSteps: self.iterationsList[0],
                sumTrainingTime: self.trainingTimeSum,
                "id": self.modelName}

    def getSplitDistrList(self, isDoorKey):
        data = []
        if self.loadedArgsDict[trainEvolutionary] and not self.canceled:
            keys = self.usedEnvEnumeration
            data = []

            for i in range(self.framesTrained // 1000000):
                split_value = f"{i + 1}m"
                row = {"trained until": split_value, "id": self.modelName}
                for key in keys:
                    row[key] = self.splitDistrBestCurric[i][key]
                data.append(row)
        return data

    def getEnvRewards(self, isEvolutionary: bool, isRRH: bool, curricLen: int):
        snapshots = []
        bestCurricScores = []
        bestGenDict = defaultdict(int)  # counts which generation had the best curricula throughout training
        env1 = []
        env2 = []
        env3 = []
        env4 = []
        if isEvolutionary:
            allCurricScores = []
            for epoch in self.rawReward:
                bestCurriculumScore = -1
                allCurricScores.append([])
                bo = self.modelName == "5_RSRun_100k_3step_3gen_3curric_nRS"
                for gen in self.rawReward[epoch]:
                    for curric in self.rawReward[epoch][gen]:
                        rewards = (self.rawReward[epoch][gen][curric])
                        currentCurricScore = 0
                        for i in range(len(rewards)):
                            currentCurricScore += np.sum(rewards[i]) * self.gamma ** i
                        allCurricScores[-1].append(currentCurricScore / self.curricMaxReward)

                        if currentCurricScore > bestCurriculumScore:
                            bestCurriculumScore = currentCurricScore
                            helper = [gen, curric, rewards]
                bestCurricScores.append(bestCurriculumScore / self.curricMaxReward)
                snapshots.append(np.sum(helper[2][0]))
                bestGenDict[helper[0]] += 1
                env1.append(sum(reward[0] for reward in helper[2]) / curricLen)
                env2.append(sum(reward[1] for reward in helper[2]) / curricLen)
                env3.append(sum(reward[2] for reward in helper[2]) / curricLen)
                env4.append(sum(reward[3] for reward in helper[2]) / curricLen)

                if bo and len(allCurricScores) == 12:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    transposed_list = list(zip(*allCurricScores))
                    sns.set_theme(style="darkgrid")
                    df = pd.DataFrame(transposed_list)
                    df = df.transpose()
                    iterationsColumn = self.iterationsList[:len(allCurricScores)]
                    df['iterationSteps'] = iterationsColumn

                    for column in df:
                        if column != "iterationSteps":
                            sns.scatterplot(data=df, x="iterationSteps", y=column, )

                    df2 = pd.DataFrame(bestCurricScores)
                    df2["iterationSteps"] = iterationsColumn
                    sns.lineplot(data=df2, x="iterationSteps", y=0)
                    plt.title("Potential Curriculum Progression", fontsize=titleFontsize)
                    plt.ylabel("Reward", fontsize=labelFontsize)
                    plt.xlabel("Iteration Steps", fontsize=labelFontsize)
                    plt.xlim(right=self.iterationsList[len(bestCurricScores) - 1] + 20000, left=0)
                    plt.xticks(fontsize=tickFontsize)
                    plt.yticks(fontsize=tickFontsize)
                    plt.tight_layout()
                    plt.show()
        elif isRRH:
            for epoch in self.rawReward:
                bestCurriculumScore = -1
                for curric in self.rawReward[epoch]:
                    rewards = (self.rawReward[epoch][curric])
                    currentCurricScore = 0
                    for i in range(len(rewards)):
                        currentCurricScore += np.sum(rewards[i]) * self.gamma ** i
                    if currentCurricScore > bestCurriculumScore:
                        bestCurriculumScore = currentCurricScore
                        helper = [curric, rewards]
                bestCurricScores.append(bestCurriculumScore)
                snapshot = np.sum(helper[1][0])
                snapshots.append(snapshot)
                env1.append(sum(reward[0] for reward in helper[1]) / curricLen)
                env2.append(sum(reward[1] for reward in helper[1]) / curricLen)
                env3.append(sum(reward[2] for reward in helper[1]) / curricLen)
                env4.append(sum(reward[3] for reward in helper[1]) / curricLen)
        else:
            for epoch in self.rawReward:
                currentReward = self.rawReward[epoch]
                env1.append(currentReward[0] / 6)  # TODO this is not accurate for dynamic obnstacle !!
                env2.append(currentReward[1] / 8)
                env3.append(currentReward[2] / 10)
                env4.append(currentReward[3] / 12)
        return env1, env2, env3, env4, bestGenDict
