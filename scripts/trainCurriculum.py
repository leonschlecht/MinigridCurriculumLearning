import multiprocessing
import sys
from datetime import datetime

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
    if not args.dynamicObstacle:
        print("Using DoorKey")
        envHintForModelName = "Door"
        envs = ENV_NAMES.DOORKEY_ENVS
    else:
        print("Using Dynamic Obstacles")
        envHintForModelName = "Obst"
        envs = ENV_NAMES.DYNAMIC_OBST_ENVS
    registerEnvs(envs, 1.0)

    if args.noRewardShaping:
        reshapingString = "_noRS"
    else:
        reshapingString = "_RS"
    constMaxsteps = ""
    if args.constMaxsteps:
        constMaxsteps = "_Constmaxstep"
    # TODO what about gamma and Crossover / Mutation rates & evolAlgo in the modelname?
    model = args.model + "_" + envHintForModelName + reshapingString + constMaxsteps + "_s" + str(args.seed)
    # get txt logger creates the directory
    txtLogger = utils.get_txt_logger(utils.get_model_dir(model))
    startTime: datetime = datetime.now()
    ############
    assert args.stepsPerCurric > 0, "Steps per curriculum must be >= 1"
    assert args.numCurric > 0, "There must be at least 1 curriculum"
    assert args.iterPerEnv > 0, "The iterations per curricululm step must be >= 1"
    assert 0 < args.paraEnv <= len(
        ENV_NAMES.DOORKEY_ENVS), "Cant train on more envs in parallel than there are envs available"

    if args.trainEvolutionary:
        e = RollingHorizonEvolutionaryAlgorithm(txtLogger, startTime, cmdLineString, args, model)
        e.startCurriculumTraining()
    elif args.trainRandomRH:
        e = RandomRollingHorizon(txtLogger, startTime, cmdLineString, args, model)
        e.startCurriculumTraining()
    elif args.trainAllParalell:
        e = allParalell(txtLogger, startTime, cmdLineString, args, model)
    else:
        raise Exception("No training method selected!")


if __name__ == "__main__":
    main()
