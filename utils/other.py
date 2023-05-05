import random
import numpy
import torch
import collections

from utils import ENV_NAMES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def getEnvListThroughDifficulty(difficulty: int) -> list:
    envList = []
    for i in range(len(ENV_NAMES.ALL_ENVS)):
        envList.append(getEnvFromDifficulty(i, difficulty))
    return envList


def getEnvFromDifficulty(index: int, envDifficulty) -> str:
    return ENV_NAMES.ALL_ENVS[index] + ENV_NAMES.CUSTOM_POSTFIX + str(envDifficulty)
