import random
import numpy
import torch
import collections

from utils import ENV_NAMES

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed(randomSeed):
    random.seed(randomSeed)
    numpy.random.seed(randomSeed)
    torch.manual_seed(randomSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(randomSeed)


def synthesize(array):
    d = collections.OrderedDict()
    d["mean"] = numpy.mean(array)
    d["std"] = numpy.std(array)
    d["min"] = numpy.amin(array)
    d["max"] = numpy.amax(array)
    return d


def getEnvListThroughDifficulty(difficulty: int, rawEnvList: list) -> list:
    envList = []
    for i in range(len(rawEnvList)):
        envList.append(getEnvFromDifficulty(i, rawEnvList, difficulty))
    return envList


def getEnvFromDifficulty(index: int, envList: list, envDifficulty) -> str:
    return envList[index] + ENV_NAMES.CUSTOM_POSTFIX + str(envDifficulty)
