import os
import random
from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray

import utils
from curricula import train, evaluate
from utils import ENV_NAMES, getEnvFromDifficulty
from utils import getModelWithCandidatePrefix
from utils.curriculumHelper import *


class allParalell:
    def __init__(self, txtLogger, startTime, cmdLineString: str, args):
        # random.seed(args.seed)
        self.args = args
        self.numCurric = args.numCurric
        self.stepsPerCurric = args.stepsPerCurric
        self.cmdLineString = cmdLineString
        self.lastEpochStartTime = startTime
        self.envDifficulty = 0
        self.exactIterationsSet = False
        self.seed = args.seed
        self.paraEnvs = args.paraEnv

        self.ITERATIONS_PER_ENV = 500000
        self.iterationsDone = 0
        self.txtLogger = txtLogger
        self.selectedModel = utils.getEpochModelName(args.model, 0)
        self.totalEpochs = args.trainEpochs
        self.trainingTime = 0

        self.stepMaxReward = calculateCurricStepMaxReward(len(ENV_NAMES.ALL_ENVS))
        self.curricMaxReward = calculateCurricMaxReward(self.stepsPerCurric, self.stepMaxReward, args.gamma)

        self.trainingInfoJson = {}
        self.logFilePath = os.getcwd() + "\\storage\\" + args.model + "\\status.json"  # TODO maybe outsource
        self.gamma = args.gamma  # TODO is gamma used properly? Do RH -> Get Max thingy, and update difficulty based on the RH reward or snapshot reward?
        self.currentRewardsDict = {}
        self.currentSnapshotRewards = {}
        self.curriculaEnvDetails = {}
        self.model = args.model
        self.modelExists = os.path.exists(self.logFilePath)
        self.trainEachCurriculum(self.totalEpochs)

    def trainEachCurriculum(self, totalEpochs):
        # TODO create logfiles etc
        # TODO selectedModel shouldnt be epoch0

        envNames = self.updateEnvNames(0)

        iterationsDone = 0
        for epoch in range(totalEpochs):
            iterationsDone = train.startTraining(iterationsDone + self.ITERATIONS_PER_ENV, iterationsDone,
                                                 self.selectedModel, envNames, self.args, self.txtLogger)
            reward = evaluate.evaluateAgent(self.selectedModel, self.envDifficulty, self.args, self.txtLogger)
            envNames = self.updateEnvNames(reward)
            self.txtLogger.info(f"reward {reward}")

    def updateEnvNames(self, currentReward) -> list:
        envNames = []
        maxReward = self.stepMaxReward
        print(maxReward)
        if currentReward > maxReward * .75:
            difficulty = 2
        elif currentReward > maxReward * .25:
            difficulty = 1
        else:
            difficulty = 2
        for j in range(len(ENV_NAMES.ALL_ENVS)):
            index = j
            if j == 3:
                index = 0
            envNames.append(getEnvFromDifficulty(index, difficulty))
        print(envNames) # TODO fix allFrames bug!!! This should be loaded and NOT be set manually
        return envNames
