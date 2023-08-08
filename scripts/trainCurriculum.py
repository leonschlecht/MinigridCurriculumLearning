import multiprocessing
import sys
from datetime import datetime

from gymnasium.envs.registration import register

import utils
from curricula import RollingHorizonEvolutionaryAlgorithm, RandomRollingHorizon, allParalell
from utils import ENV_NAMES
from utils.curriculumHelper import registerEnvs


def main():
    # set fork for linux
    try:
        multiprocessing.get_context("fork")
    except:
        print("fork not set")
    cmdLineString = ' '.join(sys.argv)
    args = utils.initializeArgParser()
    # TODO add --debug option with some preset parameters, and only use more params if != default ones (+ rnd model name)

    if not args.dynamicObstacle:
        print("Using DoorKey")
        envs = ENV_NAMES.DOORKEY_ENVS
    else:
        print("Using Dynamic Obstacles")
        envs = ENV_NAMES.DYNAMIC_OBST_ENVS
    registerEnvs(envs, 1.0)

    # TODO this is not clear if it creates a folder or not
    model = args.model + "_s" + str(args.seed)
    txtLogger = utils.get_txt_logger(utils.get_model_dir(model))
    startTime: datetime = datetime.now()

    ############
    assert args.stepsPerCurric > 0, "Steps per curriculum must be >= 1"
    assert args.numCurric > 0, "There must be at least 1 curriculum"
    assert args.iterPerEnv > 0, "The iterations per curricululm step must be >= 1"
    assert 0 < args.paraEnv <= len(
        ENV_NAMES.DOORKEY_ENVS), "Cant train on more envs in parallel than there are envs available"

    if args.trainEvolutionary:
        e = RollingHorizonEvolutionaryAlgorithm(txtLogger, startTime, cmdLineString, args)
        e.startCurriculumTraining()
    elif args.trainRandomRH:
        e = RandomRollingHorizon(txtLogger, startTime, cmdLineString, args, True) # TODO remove True param
        e.startCurriculumTraining()
    elif args.trainAllParalell:
        e = allParalell(txtLogger, startTime, cmdLineString, args)
    else:
        raise Exception("No training method selected!")


if __name__ == "__main__":
    main()
