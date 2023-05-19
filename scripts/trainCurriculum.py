import sys
from datetime import datetime

import numpy as np
from gymnasium.envs.registration import register

import utils
from curricula import linearCurriculum, RollingHorizonEvolutionaryAlgorithm, \
    adaptiveCurriculum, RandomRollingHorizon
from utils import ENV_NAMES


def main():
    cmdLineString = ' '.join(sys.argv)
    args = utils.initializeArgParser()
    # TODO add --debug option with some preset parameters, and only use more params if != default ones (+ rnd model name)

    # TODO this is not clear if it creates a folder or not
    txtLogger = utils.get_txt_logger(utils.get_model_dir(args.model))
    startTime: datetime = datetime.now()

    ############
    # TODO make e.start() methods instaed of doing it in init because of calling eval later
    assert args.stepsPerCurric > 0, "Steps per curriculum must be >= 1"
    assert args.numCurric > 0, "There must be at least 1 curriculum"
    assert args.iterPerEnv > 0, "The iterations per curricululm step must be >= 1"
    assert args.trainEpochs > 1, "There must be at least 2 training epochs for the algorithm"
    assert 0 < args.paraEnv <= len(ENV_NAMES.ALL_ENVS)

    if args.trainEvolutionary:
        e = RollingHorizonEvolutionaryAlgorithm(txtLogger, startTime, cmdLineString, args)
    elif args.trainBiasedRandomRH:
        e = RandomRollingHorizon(txtLogger, startTime, cmdLineString, args, False)
    elif args.trainRandomRH:
        e = RandomRollingHorizon(txtLogger, startTime, cmdLineString, args, True)
    elif args.trainLinear:
        linearCurriculum.startLinearCurriculum(txtLogger, startTime, args)
    elif args.trainAdaptive:
        adaptiveCurriculum.startAdaptiveCurriculum(txtLogger, startTime, args)
    else:
        print("No training method selected!")


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
    ENV_SIZE_POWER = 2
    SIZE_MUTIPLICATOR = 10
    maxStepsEnv4 = 12 ** ENV_SIZE_POWER * SIZE_MUTIPLICATOR
    maxStepsEnv3 = 9 ** ENV_SIZE_POWER * SIZE_MUTIPLICATOR
    maxStepsEnv2 = 7 ** ENV_SIZE_POWER * SIZE_MUTIPLICATOR
    maxStepsEnv1 = 4 ** ENV_SIZE_POWER * SIZE_MUTIPLICATOR
    maxSteps = np.array([maxStepsEnv1, maxStepsEnv2, maxStepsEnv3, maxStepsEnv4])
    difficulty = np.array([1, 0.33, 0.11])
    result = np.round(np.matmul(maxSteps.reshape(-1, 1), difficulty.reshape(1, -1)))

    for i in range(len(difficulty)):
        register(
            id=ENV_NAMES.DOORKEY_12x12 + ENV_NAMES.CUSTOM_POSTFIX + str(i),
            entry_point="minigrid.envs:DoorKeyEnv",
            kwargs={"size": 12, "max_steps": int(result[3][i])},
        )

        register(
            id=ENV_NAMES.DOORKEY_9x9 + ENV_NAMES.CUSTOM_POSTFIX + str(i),
            entry_point="minigrid.envs:DoorKeyEnv",
            kwargs={"size": 9, "max_steps": int(result[2][i])},
        )

        register(
            id=ENV_NAMES.DOORKEY_7x7 + ENV_NAMES.CUSTOM_POSTFIX + str(i),
            entry_point="minigrid.envs:DoorKeyEnv",
            kwargs={"size": 7, "max_steps": int(result[1][i])},
        )

        register(
            id=ENV_NAMES.DOORKEY_4x4 + ENV_NAMES.CUSTOM_POSTFIX + str(i),
            entry_point="minigrid.envs:DoorKeyEnv",
            kwargs={"size": 4, "max_steps": int(result[0][i])},
        )


if __name__ == "__main__":
    registerEnvs()
    main()
