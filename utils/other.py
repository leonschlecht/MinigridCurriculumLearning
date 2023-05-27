import random
import numpy
import torch
import collections

from utils import ENV_NAMES

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""
print('Using device:', device)
x = torch.zeros((10, 10), device=device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    print(torch.cuda.device_count())

t = torch.cuda.get_device_properties(device).total_memory
r = torch.cuda.memory_reserved(device)
a = torch.cuda.memory_allocated(device)
f = r-a  # free inside reserved
from pynvml import *
nvmlInit()
h = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(h)
print(f'total    : {info.total}')
print(f'free     : {info.free}')
print(f'used     : {info.used}')
with torch.cuda.device(device):
    info = torch.cuda.mem_get_info()
print(f'free     : {info.free/1024**3:.1f} GB')
print(f'total    : {info.total/1024**3:.1f} GB')
"""
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
