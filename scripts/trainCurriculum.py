import multiprocessing
import sys
from datetime import datetime

import numpy as np
from gymnasium.envs.registration import register

import utils
from curricula import linearCurriculum, RollingHorizonEvolutionaryAlgorithm, \
    adaptiveCurriculum, RandomRollingHorizon, allParalell
from utils import ENV_NAMES
from utils.curriculumHelper import maxStepsEnv4, maxStepsEnv3, maxStepsEnv2, maxStepsEnv1


def main():
    try:
        multiprocessing.get_context("fork")
    except:
        print("fork not set")
    cmdLineString = ' '.join(sys.argv)
    args = utils.initializeArgParser()
    # TODO add --debug option with some preset parameters, and only use more params if != default ones (+ rnd model name)

    # TODO this is not clear if it creates a folder or not
    model = args.model + "_s" + str(args.seed)
    txtLogger = utils.get_txt_logger(utils.get_model_dir(model))
    startTime: datetime = datetime.now()

    ############
    assert args.stepsPerCurric > 0, "Steps per curriculum must be >= 1"
    assert args.numCurric > 0, "There must be at least 1 curriculum"
    assert args.iterPerEnv > 0, "The iterations per curricululm step must be >= 1"
    assert args.trainEpochs > 1, "There must be at least 2 training epochs for the algorithm"
    assert 0 < args.paraEnv <= len(
        ENV_NAMES.ALL_ENVS), "Cant train on more envs in parallel than there are envs available"

    if args.trainEvolutionary:
        e = RollingHorizonEvolutionaryAlgorithm(txtLogger, startTime, cmdLineString, args)
        e.startCurriculumTraining()
    elif args.trainBiasedRandomRH:
        e = RandomRollingHorizon(txtLogger, startTime, cmdLineString, args, False)
        e.startCurriculumTraining()
    elif args.trainRandomRH:
        e = RandomRollingHorizon(txtLogger, startTime, cmdLineString, args, True)
        e.startCurriculumTraining()
    elif args.trainLinear:
        linearCurriculum.startLinearCurriculum(txtLogger, startTime, args)
    elif args.trainAllParalell:
        e = allParalell(txtLogger, startTime, cmdLineString, args)
    elif args.trainAdaptive:
        adaptiveCurriculum.startAdaptiveCurriculum(txtLogger, startTime, args)
    else:
        raise Exception("No training method selected!")


def registerEnvs():
    """
    Registers the envs before the training. Each env has 3 difficulty settings, with which they decrease their maxsteps
    to save computation time and make the environment harder
    """
    """
    register(
        id="MiniGrid-Empty-Random-8x8-v0",
        entry_point="minigrid.envs:EmptyEnv",
        kwargs={"size": 8, "agent_start_pos": None},
    ) """
    # Register Default envs at normal difficulty
    s = "1.0"
    register(
        id=ENV_NAMES.DOORKEY_12x12 + ENV_NAMES.CUSTOM_POSTFIX + s,
        entry_point="minigrid.envs:DoorKeyEnv",
        kwargs={"size": 12, "max_steps": int(maxStepsEnv4)},
    )
    register(
        id=ENV_NAMES.DOORKEY_10x10 + ENV_NAMES.CUSTOM_POSTFIX + s,
        entry_point="minigrid.envs:DoorKeyEnv",
        kwargs={"size": 10, "max_steps": int(maxStepsEnv3)},
    )

    register(
        id=ENV_NAMES.DOORKEY_8x8 + ENV_NAMES.CUSTOM_POSTFIX + s,
        entry_point="minigrid.envs:DoorKeyEnv",
        kwargs={"size": 8, "max_steps": int(maxStepsEnv2)},
    )

    register(
        id=ENV_NAMES.DOORKEY_6x6 + ENV_NAMES.CUSTOM_POSTFIX + s,
        entry_point="minigrid.envs:DoorKeyEnv",
        kwargs={"size": 6, "max_steps": int(maxStepsEnv1)},
    )


if __name__ == "__main__":
    registerEnvs()
    main()
